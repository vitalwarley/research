import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt


def extract_loss_auc(file_path, loss, metric):
    losses = []
    metrics = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if loss in line:
            loss_val = float(re.search(rf"{loss}:(\d+\.\d+)", line).group(1))
            losses.append(loss_val)
        if f"{metric} is" in line:
            metric_val = float(re.search(rf"{metric} is (\d+\.\d+)", line).group(1))
            metrics.append(metric_val)

    print(f"Number of metrics and losses: {len(metrics)}, {len(losses)}")

    return losses, metrics


def plot_metrics(epochs, losses, aucs, loss_name, metric_name, save_path=None):
    plt.figure(figsize=(12, 6))

    # get the min loss and max metric
    min_loss = min(losses)
    max_metric = max(aucs)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker="o")
    plt.plot(epochs[losses.index(min_loss)], min_loss, "ro")
    plt.legend([f"{loss_name} (min {min_loss})"], loc="upper right")
    plt.title(f"{loss_name} vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, aucs, marker="o")
    plt.plot(epochs[aucs.index(max_metric)], max_metric, "ro")
    plt.legend([f"{metric_name} (max {max_metric})"], loc="lower right")
    plt.title(f"{metric_name} vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file-path", type=str, default="input.txt")
    parser.add_argument("--save-path", type=str, default="Track1.png")
    # add loss and metric arguments
    parser.add_argument("--loss", type=str, default="contrastive_loss")
    parser.add_argument("--metric", type=str, default="auc")
    args = parser.parse_args()
    losses, metrics = extract_loss_auc(args.file_path, args.loss, args.metric)
    epochs = list(range(1, len(losses) + 1))
    plot_metrics(epochs, losses, metrics, args.loss, args.metric, args.save_path)
