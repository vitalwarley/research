import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")


def load_data(file_path):
    df = pd.read_csv(file_path)
    df["normalize"] = df["normalize"].astype("category")
    return df


def parse_flag(flag):
    # Using regular expression to match field, operator, and value
    match = re.match(r"([a-zA-Z0-9_-]+)([<>=!]+)([\d.]+)", flag)
    if match:
        field, operator, value = match.groups()
        return field, operator, value
    else:
        raise ValueError(f"Invalid flag format: {flag}")


def parse_flags(flags):
    parsed_flags = {}
    for flag in flags:
        field, operator, value = parse_flag(flag)
        parsed_flags[field] = f"{operator} {value}"
    return parsed_flags


def filter_data(df, metric, metric_value, step_range, flags):
    max_step = df.step.max() if step_range[1] == -1 else step_range[1]
    _, op, value = parse_flag(f"{metric}{metric_value}")
    query = f"(metric == '{metric}') & (value {op} {value}) & ({step_range[0]} < step) & (step < {max_step})"
    for flag, condition in flags.items():
        query += f" & (`{flag}` {condition})"
    print(f"Query: {query}")
    return df.query(query)


def get_top_experiments(df, n):
    return df.groupby("experiment")[["step", "value"]].max("value").sort_values(by="value", ascending=False).head(n)


def process_experiments(df, top_n_experiments):
    print(f"Top {top_n_experiments.shape[0]} experiments from {df.experiment.unique().shape[0]} total experiments.")
    experiments = []
    steps = top_n_experiments.step.unique()
    for experiment in top_n_experiments.index:
        exp = df[(df["experiment"] == experiment) & (df.step.isin(steps))][
            ["experiment", "step", "batch-size", "clip-gradient", "normalize", "num-epoch", "metric", "value"]
        ]
        exp["experiment"] = experiment
        # slice row with max "value"
        exp = exp.loc[exp.groupby(["experiment", "metric"])["value"].idxmax()]
        experiments.append(exp)

    experiments_df = pd.concat(experiments)
    print(f"Remaining experiments: {experiments_df.experiment.unique().shape[0]}")
    print(experiments_df)
    return experiments_df


def plot_metrics_vs_epoch_auc(metric, experiments, output_dir, title):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(10, 5), sharey=True)
    # axes = axes.flatten()
    # Set a color for each experiment
    cmap = plt.get_cmap("tab10")
    for idx, col in enumerate(["batch-size", "clip-gradient", "normalize", "num-epoch"]):
        metric_df = experiments[experiments.metric == metric]
        # Add color
        # axes[idx].scatter(metric_df[col], metric_df["value"])
        for idx_exp, experiment in enumerate(metric_df.experiment.unique()):
            exp_df = metric_df[metric_df.experiment == experiment]
            axes[idx].scatter(
                exp_df[col], exp_df["value"], label=experiment, color=cmap(idx_exp), marker=".", alpha=0.9
            )
        axes[idx].set_xlabel(col)
        # axes[idx].set_ylabel(metric)
        # axes[idx].set_title(f"{col} vs {metric}")
        if col == "normalize":
            axes[idx].set_xticks([0, 1])
            axes[idx].set_xticklabels(["No", "Yes"])
        # If axes are shared, only set the ylabel for the leftmost column
        if idx == 0:
            axes[idx].set_ylabel(metric)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"{output_dir}/{metric}_vs_hparams_{now}.png")
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze TensorBoard data")
    parser.add_argument("--file", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--metric", type=str, required=True, help="Metric to analyze")
    parser.add_argument("--metric-value", type=str, required=True, help="Metric value to analyze")
    parser.add_argument("--start-step", type=int, default=0, help="Start step for filtering")
    parser.add_argument("--end-step", type=int, default=-1, help="End step for filtering")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top experiments to analyze")
    parser.add_argument("--flags", nargs="*", help="Flags to filter data in the format flag=value")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_arguments()

    df = load_data(args.file)

    # Parse flags into a dictionary
    flags = {}
    if args.flags:
        flags = parse_flags(args.flags)
    print(f"Flags: {flags}")

    filtered_df = filter_data(df, args.metric, args.metric_value, [args.start_step, args.end_step], flags)
    top_n_experiments = get_top_experiments(filtered_df, args.top_n)
    experiments = process_experiments(df, top_n_experiments)
    n_experiments = experiments.experiment.unique().shape[0]
    title = (
        f"{n_experiments} experiment for {args.metric} ({args.metric_value})\nStep {args.start_step} to {args.end_step}"
    )
    if flags:
        title += f"\nwith Flags {flags}"
    plot_metrics_vs_epoch_auc(args.metric, experiments, args.output_dir, title)


if __name__ == "__main__":
    main()
