import os
import pickle
import shutil
from pathlib import Path
from typing import List, Tuple, Callable

import cv2
import numpy as np
import mxnet as mx
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from more_itertools import grouper

import mytypes as t
from dataset import FamiliesDataset, PairDataset, ImgDataset


def norm(emb):
    return np.sqrt(np.sum(emb**2))


def cosine(emb1, emb2):
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))


def euclidean(emb1, emb2):
    return -norm(emb1 - emb2)


def load_lfw(root: str, target: str, _image_size):
    bin_path = Path(root) / target
    # read bin
    with open(bin_path, "rb") as f:
        bins, labels = pickle.load(f, encoding="bytes")
    for idx, (first, second) in enumerate(grouper(bins, 2)):
        if first is None or second is None:
            continue
        first = cv2.imdecode(first, cv2.IMREAD_COLOR)
        first = cv2.resize(first, _image_size)
        first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
        second = cv2.imdecode(second, cv2.IMREAD_COLOR)
        second = cv2.resize(second, _image_size)
        second = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)
        yield (first, second), labels[idx]


def predict_on_lfw(model):
    similarities = []
    labels = []
    for (first, second), label in tqdm(
        load_lfw("/home/warley/dev/datasets/MS1M_v3", "lfw.bin", (112, 112)), total=6000
    ):
        similarity = model(first, second)
        similarities.append(similarity)
        labels.append(label)
    return np.stack(similarities, axis=0), np.stack(labels, axis=0)


def predict(
    model: Callable[[Path, Path], t.Labels], pair_list: List[t.PairPath]
) -> t.Labels:
    predictions = []
    for idx, (path1, path2) in tqdm(enumerate(pair_list), total=len(pair_list)):
        cur_prediction = model(path1, path2)
        predictions.append(cur_prediction)
    return np.stack(predictions, axis=0)


class CompareModelONNX(object):
    def __init__(self, model_name: str = "model.onnx"):
        model_name = str(model_name)
        import onnxruntime as ort

        EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_name, providers=EP_list)
        self.embeddings = dict()
        self.metric = cosine
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def get_embedding(self, im_path: Path) -> t.Embedding:
        if isinstance(im_path, Path):
            img = cv2.imread(str(im_path))
        else:
            img = im_path
        img = img.transpose(2, 1, 0).reshape(1, 3, 112, 112).astype(np.float32)
        # img = ((img / 255.0) - 0.5) / 0.5
        return self.session.run([self.output_name], {self.input_name: img})[0].reshape(
            -1,
        )

    def __call__(self, path1: Path, path2: Path) -> t.Labels:
        emb1 = self.get_embedding(path1)
        emb2 = self.get_embedding(path2)
        return self.metric(emb1, emb2)


class CompareModel(object):
    def __init__(self, model_name: str = "arcface_r100_v1", ctx: mx.Context = mx.cpu()):
        root = Path("fitw2020") / "models"
        model_name = str(root / model_name)
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, 0)
        sym = sym.get_internals()["fc1_output"]
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        data_shape = (1, 3, 112, 112)
        model.bind(data_shapes=[("data", data_shape)], for_training=False)
        model.set_params(arg_params, aux_params)
        # warmup
        data = mx.nd.zeros(shape=data_shape)
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        _ = model.get_outputs()[0].asnumpy()
        self.model = model
        self.embeddings = dict()
        self.metric = cosine

    def get_embedding(self, im_path: Path) -> t.Embedding:
        if isinstance(im_path, Path):
            img = (
                mx.img.imread(str(im_path))
                .transpose((2, 0, 1))
                .expand_dims(0)
                .astype(np.float32)
            )
        else:
            img = (
                mx.nd.array(im_path)
                .transpose((2, 0, 1))
                .expand_dims(0)
                .astype(np.float32)
            )
        batch = mx.io.DataBatch([img])
        self.model.forward(batch, is_train=False)
        return self.model.get_outputs()[0][0].asnumpy()

    def __call__(self, path1: Path, path2: Path) -> t.Labels:
        emb1 = self.get_embedding(path1)
        emb2 = self.get_embedding(path2)
        return self.metric(emb1, emb2)


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
    y_true = [label for _, _, label in pairs]
    pair_list = [(img1, img2) for img1, img2, _ in pairs]
    return pair_list, y_true


if __name__ == "__main__":
    print(mx.__version__)
    pair_list, y_true = load_pairs()

    model = CompareModelONNX(model_name="arcface_r100_v1.onnx")
    model.metric = cosine
    predictions, y_true = predict_on_lfw(model)
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(
        fpr,
        tpr,
        color="b",
        lw=1,
        label=f"lfw AUC (insightface mxnet):{auc(fpr, tpr):.4f}",
    )

    model = CompareModelONNX(model_name="my_pretrained_model.onnx")
    model.metric = cosine
    predictions, y_true = predict_on_lfw(model)
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(
        fpr, tpr, color="g", lw=1, label=f"lfw AUC (my torch model):{auc(fpr, tpr):.4f}"
    )

    plt.xlabel("Flase Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (using ONNX models)")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("roc.pdf", transparent=True, pad_inches=0, bbox_inches="tight")
    plt.show()
