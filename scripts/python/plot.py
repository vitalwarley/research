import re
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_log_file(file_path):
    steps, lrs, losses, accuracies = [], [], [], []
    epoch_step_dict = {}
    epoch_loss_dict = {}
    epoch_acc_dict = {}

    with open(file_path, "r") as file:
        for line in file:
            pattern = re.compile(
                r"Epoch\s+(\d+)\s+\|\s+Step\s+(\d+)\s+-\s+Loss:\s+(\d+\.\d+),\s+Acc:\s+(\d+\.\d+),\s+LR:\s+(\d+\.\d+)"
            )
            match = pattern.match(line.strip())
            if match:
                epoch_num = int(match.group(1))
                step_num = int(match.group(2))
                loss_value = float(match.group(3))
                acc_value = float(match.group(4))
                lr_value = float(match.group(5))

                steps.append(step_num)
                lrs.append(lr_value)
                losses.append(loss_value)
                accuracies.append(acc_value)

                epoch_step_dict.setdefault(epoch_num, []).append(step_num)
                epoch_loss_dict.setdefault(epoch_num, []).append(loss_value)
                epoch_acc_dict.setdefault(epoch_num, []).append(acc_value)

    return steps, lrs, losses, accuracies, epoch_step_dict, epoch_loss_dict, epoch_acc_dict


def plot_metric(ax, steps, values, metric_name, ylabel, epoch_step_dict, epoch_metric_dict):
    ax.plot(steps, values, label=metric_name, color="blue")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric_name} by Step")
    ax.grid(True)
    ax.legend()

    # Calculate and plot the mean for each epoch
    epoch_means = [np.mean(epoch_metric_dict[epoch]) for epoch in sorted(epoch_metric_dict)]
    epoch_steps = [epoch_step_dict[epoch][-1] for epoch in sorted(epoch_step_dict)]  # Get last step of each epoch
    ax.plot(epoch_steps, epoch_means, "r--", label="Epoch Mean")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file-path", type=str, required=True)
    args = parser.parse_args()
    logs_dir = Path(args.file_path).parent

    steps, lrs, losses, accuracies, epoch_step_dict, epoch_loss_dict, epoch_acc_dict = parse_log_file(args.file_path)

    sns.set(style="whitegrid")  # Applying seaborn style

    fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

    # Plot Learning Rate
    plot_metric(axes[0], steps, lrs, "Learning Rate", "Learning Rate", {}, {})
    axes[0].label_outer()  # Hides x-axis labels

    # Plot Loss
    plot_metric(axes[1], steps, losses, "Loss", "Loss", epoch_step_dict, epoch_loss_dict)
    axes[1].label_outer()  # Hides x-axis labels

    # Plot Accuracy
    plot_metric(axes[2], steps, accuracies, "Accuracy", "Accuracy", epoch_step_dict, epoch_acc_dict)

    plt.tight_layout()
    plt.savefig(logs_dir / "plot.png")
    plt.show()
