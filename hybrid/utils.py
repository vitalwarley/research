import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from more_itertools import grouper
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import KFold


def plot_roc(tpr, fpr, savedir=None):
    data = pd.DataFrame(dict(tpr=tpr, fpr=fpr))
    p = sns.lineplot(data=data, x="fpr", y="tpr")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    fig = p.get_figure()
    if savedir is not None:
        fig.savefig(Path(savedir) / "roc.png")
    plt.close(fig)
    return fig


def load_pairs():
    pairs = []
    root_path = Path("/home/warley/dev/datasets/fiw/val-faces-det")
    with open("/home/warley/dev/datasets/fiw/val_pairs.csv", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue
            img1, img2, label = line.split(",")
            pairs.append((root_path / img1, root_path / img2, int(label)))
    y_true = np.array([label for _, _, label in pairs])
    pair_list = np.array([(img1, img2) for img1, img2, _ in pairs])
    return pair_list, y_true


def load_lfw(root: str, target: str, _image_size):
    bin_path = Path(root) / target
    # read bin
    with open(bin_path, "rb") as f:
        bins, labels = pickle.load(f, encoding="bytes")
    for idx, (first, second) in enumerate(grouper(bins, 2)):
        if first is None or second is None:
            continue
        first = cv2.imdecode(first, cv2.IMREAD_COLOR)
        first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
        first = cv2.resize(first, _image_size)
        second = cv2.imdecode(second, cv2.IMREAD_COLOR)
        second = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)
        second = cv2.resize(second, _image_size)
        # in EvalPretrainDataset I return the scaled image,
        # but here I return the original image.
        # For my onnx, I perform scaling on get_embeddings
        # for the mxnet, we don't need scaling.
        yield (first, second), labels[idx]


def compute_kfold_on(values, y_true, k=10, fn=None):
    """
    Compute kfold on values (distances or similarities).
    """
    assert fn is not None
    kfold = KFold(n_splits=k, shuffle=False)
    n_pairs = values.shape[0]
    max_distance = np.max(values)
    accuracy = np.zeros((k), dtype=np.float32)
    # i think i got this end=4 from insightface, which makes sense with clip_grad_val.
    thresholds = np.arange(0.0, max_distance, 0.01, dtype=np.float32)
    best_thresholds = np.zeros((k), dtype=np.float32)
    indexes = np.arange(n_pairs, dtype=np.int32)
    acc_train = np.zeros((thresholds.shape[0]), dtype=np.float32)

    # iterate over folds
    for fold, (train_idx, test_idx) in enumerate(kfold.split(indexes)):
        # slice train dist and labels
        train_distances = values[train_idx]
        train_labels = y_true[train_idx]
        # compute train accuracy
        for t_idx, threshold in enumerate(thresholds):
            predicted = fn(train_distances, threshold)
            acc_train[t_idx] = accuracy_score(train_labels, predicted)
        # compute test accuracy
        test_distances = values[test_idx]
        test_labels = y_true[test_idx]
        best_threshold_index = np.argmax(acc_train)
        threshold = thresholds[best_threshold_index]
        predicted = test_distances < threshold
        accuracy[fold] = accuracy_score(test_labels, predicted)
        best_thresholds[fold] = threshold

    return best_thresholds, accuracy


def make_accuracy_vs_threshold_plot(values, y_true, fn):
    # build a accuracies list for each threshold
    accuracies = []
    max_thresh = max(values)
    thresholds = np.arange(0.0, max_thresh, 0.01, dtype=np.float32)
    for threshold in thresholds:
        predicted = fn(values, threshold)
        accuracy = accuracy_score(y_true, predicted)
        accuracies.append(accuracy)

    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]

    df = pd.DataFrame({"threshold": thresholds, "accuracy": accuracies})
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig = sns.lineplot(x="threshold", y="accuracy", data=df, ax=ax).get_figure()
    ax.plot(best_threshold, best_accuracy, "o", color="red")
    ax.annotate(f"({best_threshold:.2f},{best_accuracy:.2f})", (best_threshold, best_accuracy))
    return fig, best_threshold, best_accuracy


def make_roc_plot(values, y_true):
    fpr, tpr, _ = roc_curve(y_true, values)
    auc_score = auc(fpr, tpr)

    df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig = sns.lineplot(x="fpr", y="tpr", data=df, ax=ax).get_figure()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    return fig, auc_score


def log_results(
    writer,
    base_tag,
    distances,
    similarities,
    y_true,
    global_step: int = -1,
    log_auc: bool = True,
):
    positive_samples = y_true == 1
    negative_samples = y_true == 0

    if any(positive_samples):
        writer.add_histogram(
            f"{base_tag}/distances/positive",
            distances[positive_samples],
            global_step=global_step if global_step >= 0 else 1,
        )
        writer.add_histogram(
            f"{base_tag}/similarities/positive",
            similarities[positive_samples],
            global_step=global_step if global_step >= 0 else 1,
        )
    if any(negative_samples):
        writer.add_histogram(
            f"{base_tag}/distances/negative",
            distances[negative_samples],
            global_step=global_step if global_step >= 0 else 0,
        )
        writer.add_histogram(
            f"{base_tag}/similarities/negative",
            similarities[negative_samples],
            global_step=global_step if global_step >= 0 else 0,
        )

    fig, *_ = make_accuracy_vs_threshold_plot(distances, y_true, fn=lambda x, thresh: x < thresh)
    writer.add_figure(
        f"{base_tag}/distances/accuracy vs threshold",
        fig,
        global_step=global_step if global_step >= 0 else 0,
    )
    fig, best_threshold, best_accuracy = make_accuracy_vs_threshold_plot(
        similarities, y_true, fn=lambda x, thresh: x > thresh
    )  # high sim, low dist
    writer.add_figure(
        f"{base_tag}/similarities/accuracy vs threshold",
        fig,
        global_step=global_step if global_step >= 0 else 0,
    )

    fig, auc_score = make_roc_plot(similarities, y_true)
    if not np.isnan(auc_score):
        if log_auc:
            writer.add_scalar(
                f"{base_tag}/roc/auc",
                auc_score,
                global_step=global_step if global_step >= 0 else 0,
            )
        writer.add_figure(
            f"{base_tag}/roc/plot",
            fig,
            global_step=global_step if global_step >= 0 else 0,
        )

    similarities[similarities <= 0] = 0  # for plotting PR curve
    writer.add_pr_curve(
        f"{base_tag}/similarities/pr curve",
        y_true,
        similarities,
        global_step=global_step if global_step >= 0 else 0,  # testing this
    )

    return best_threshold, best_accuracy, auc_score
