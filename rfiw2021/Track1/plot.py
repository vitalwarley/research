import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt


def extract_loss_auc(file_path):
    losses = []
    aucs = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if "contrastive_loss" in line:
            loss = float(re.search(r"contrastive_loss:(\d+\.\d+)", line).group(1))
            losses.append(loss)
        if "auc is" in line:
            auc = float(re.search(r"auc is (\d+\.\d+)", line).group(1))
            aucs.append(auc)

    return losses, aucs

def plot_metrics(epochs, losses, aucs, save_path=None):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o')
    plt.title('Contrastive Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Contrastive Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, aucs, marker='o')
    plt.title('AUC vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file-path", type=str, default="input.txt")
    parser.add_argument("--save-path", type=str, default="Track1.png")
    args = parser.parse_args()
    losses, aucs = extract_loss_auc(args.file_path)
    epochs = list(range(1, len(losses) + 1))
    plot_metrics(epochs, losses, aucs, args.save_path)
