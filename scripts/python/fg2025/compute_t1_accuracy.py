import subprocess
from io import StringIO
import pandas as pd
import argparse
import shlex

# Configuration order constant
CONFIG_ORDER = [
    "BB",
    "SS",
    "SIBS",
    "FS",
    "FD",
    "MS",
    "MD",
    "GFGD",
    "GMGD",
    "GFGS",
    "GMGS",
    "Avg",
]


def get_experiment_accuracies(guild_args: str, operation: str) -> pd.DataFrame:
    """Extract accuracy data from experiments using provided guild arguments.

    Args:
        guild_args: String containing guild arguments for filtering experiments
        operation: Either 'train' or 'test'

    Returns:
        DataFrame with experiment data
    """
    # Create base guild command
    guild_command = [
        "guild",
        "compare",
        "-Fo",
        f"{operation}",
    ]

    # Add any additional guild arguments
    if guild_args:
        guild_command.extend(shlex.split(guild_args))

    # Add output formatting arguments
    columns = ["run", ".label", "max accuracy"]
    if "test" in operation:
        # Add kinship relation columns for test
        columns.extend(
            [
                "accuracy/bb",
                "accuracy/ss",
                "accuracy/sibs",
                "accuracy/fs",
                "accuracy/fd",
                "accuracy/ms",
                "accuracy/md",
                "accuracy/gfgd",
                "accuracy/gmgd",
                "accuracy/gfgs",
                "accuracy/gmgs",
            ]
        )

    guild_command.extend(
        [
            "-cc",
            ",".join(columns),
            "--csv",
            "-",
        ]
    )

    try:
        # Run guild command and capture output
        result = subprocess.run(
            guild_command, check=True, capture_output=True, text=True
        )
        # Create DataFrame from the CSV output
        df = pd.read_csv(StringIO(result.stdout))

        # Rename columns
        column_mapping = {
            "run": "run",
            ".label": "label",
            "max accuracy": "accuracy",
            "accuracy/bb": "bb_acc",
            "accuracy/ss": "ss_acc",
            "accuracy/sibs": "sibs_acc",
            "accuracy/fs": "fs_acc",
            "accuracy/fd": "fd_acc",
            "accuracy/ms": "ms_acc",
            "accuracy/md": "md_acc",
            "accuracy/gfgd": "gfgd_acc",
            "accuracy/gmgd": "gmgd_acc",
            "accuracy/gfgs": "gfgs_acc",
            "accuracy/gmgs": "gmgs_acc",
        }
        df = df.rename(columns=column_mapping)
        return df
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute Guild command: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to process Guild data: {e}")


def compute_statistics(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """Compute accuracy statistics, optionally grouped by a column.

    Args:
        df: DataFrame containing accuracy data
        group_col: Column to group by (optional)

    Returns:
        DataFrame with statistics for each group
    """

    def get_stats(group):
        if len(group) == 0:
            return pd.Series(
                {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                    "best_run": None,
                }
            )

        best_idx = group["accuracy"].idxmax()
        return pd.Series(
            {
                "mean": group["accuracy"].mean(),
                "std": group["accuracy"].std(),
                "min": group["accuracy"].min(),
                "max": group["accuracy"].max(),
                "count": len(group),
                "best_run": group.loc[best_idx, "run"],
            }
        )

    if group_col:
        return df.groupby(group_col).apply(get_stats).reset_index()
    return pd.DataFrame([get_stats(df)])


def format_markdown_table(stats_df: pd.DataFrame) -> str:
    """Format statistics as a markdown table with mean ± std and test accuracy.

    Args:
        stats_df: DataFrame containing statistics by label

    Returns:
        Markdown formatted table string
    """
    # Sort labels to ensure consistent ordering
    stats_df = stats_df.sort_values("label")

    # Create table header
    table = "| Configuration | Val Acc (95% CI) | Test Acc |\n"
    table += "|--------------|-------------|----------|\n"

    # Add rows
    for _, row in stats_df.iterrows():
        # Format train accuracy as mean ± 1.96*std for 95% CI
        train_acc = f"{row['train_mean']*100:.2f} ± {1.96*row['train_std']*100:.2f}"
        # Format test accuracy
        test_acc = f"{row['test_accuracy']*100:.2f}"
        # Add row to table
        table += f"| {row['label']} | {train_acc} | {test_acc} |\n"

    return table


def format_latex_table(stats_df: pd.DataFrame) -> str:
    """Format statistics as a LaTeX table with mean ± std and test accuracy.

    Args:
        stats_df: DataFrame containing statistics by label

    Returns:
        LaTeX formatted table string
    """
    # Sort labels to ensure consistent ordering
    stats_df = stats_df.sort_values("label")

    # Create table header
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{lcc}\n"
    table += "\\toprule\n"
    table += "Configuration & Val Acc (95\\% CI) & Test Acc \\\\\n"
    table += "\\midrule\n"

    # Add rows
    for _, row in stats_df.iterrows():
        # Format train accuracy as mean ± 1.96*std for 95% CI
        train_acc = f"{row['train_mean']:.4f} $\\\pm$ {1.96*row['train_std']:.4f}"
        # Format test accuracy
        test_acc = f"{row['test_accuracy']:.4f}"
        # Add row to table, escaping underscores in labels
        escaped_label = row["label"].replace("_", "\\_")
        table += f"{escaped_label} & {train_acc} & {test_acc} \\\\\n"

    # Add table footer
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Model accuracy results}\n"
    table += "\\label{tab:model-accuracy}\n"
    table += "\\end{table}"

    return table


def format_test_metrics_markdown(stats_df: pd.DataFrame) -> str:
    """Format test metrics as a markdown table with kinship relations as columns.

    Args:
        stats_df: DataFrame containing statistics by label

    Returns:
        Markdown formatted table string
    """
    # Create table header
    metrics = [
        "BB",
        "SS",
        "SIBS",
        "FS",
        "FD",
        "MS",
        "MD",
        "GFGD",
        "GMGD",
        "GFGS",
        "GMGS",
        "Avg",
    ]
    header = "| Config | " + " | ".join(metrics) + " |\n"
    header += "|---------|" + "|".join(["-" * len(m) for m in metrics]) + "|\n"

    table = header

    # Add rows for each configuration
    for _, row in stats_df.iterrows():
        metrics_values = []
        for metric in metrics:
            if metric == "Avg":
                # Calculate average of all kinship accuracies for this configuration
                val = row.get("accuracy", float("nan"))
                metrics_values.append(f"{val*100:.2f}")
            else:
                col = f"{metric.lower()}_acc"
                val = row.get(col, float("nan"))
                metrics_values.append(f"{val*100:.2f}" if pd.notna(val) else "N/A")

        table += f"| {row['label']} | " + " | ".join(metrics_values) + " |\n"

    return table


def format_test_metrics_latex(stats_df: pd.DataFrame) -> str:
    """Format test metrics as a LaTeX table with kinship relations as columns.

    Args:
        stats_df: DataFrame containing statistics by label

    Returns:
        LaTeX formatted table string
    """
    metrics = [
        "BB",
        "SS",
        "SIBS",
        "FS",
        "FD",
        "MS",
        "MD",
        "GFGD",
        "GMGD",
        "GFGS",
        "GMGS",
        "Avg",
    ]

    # Create table header
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
    table += "\\toprule\n"
    table += "Config & " + " & ".join(metrics) + " \\\\\n"
    table += "\\midrule\n"

    # Add rows
    for _, row in stats_df.iterrows():
        metrics_values = []
        for metric in metrics:
            if metric == "Avg":
                # Calculate average of all kinship accuracies for this configuration
                val = row.get("accuracy", float("nan"))
                metrics_values.append(f"{val*100:.2f}")
            else:
                col = f"{metric.lower()}_acc"
                val = row.get(col, float("nan"))
                metrics_values.append(f"{val*100:.2f}" if pd.notna(val) else "N/A")

        escaped_label = row["label"].replace("_", "\\_")
        table += f"{escaped_label} & " + " & ".join(metrics_values) + " \\\\\n"

    # Add table footer
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Test metrics by kinship relation}\n"
    table += "\\label{tab:test-metrics-kinship}\n"
    table += "\\end{table}"

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy statistics from Guild experiments"
    )
    parser.add_argument(
        "-g",
        "--guild-args",
        type=str,
        default="",
        help="Guild arguments for filtering experiments (e.g. '-r 1:100')",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="scl",
        help="Model to compute statistics for",
    )
    args = parser.parse_args()

    # Get experiment data for train and test
    train_df = get_experiment_accuracies(args.guild_args, f"{args.model}:train")
    test_df = get_experiment_accuracies(args.guild_args, f"{args.model}:test")

    # Print raw data
    print("\nRaw Train Data:")
    print(train_df)
    print("\nRaw Test Data:")
    print(test_df)

    # Compute statistics by label for train
    train_stats = compute_statistics(train_df, "label")
    train_stats.columns = ["label"] + [
        f"train_{col}" for col in train_stats.columns if col != "label"
    ]

    # Get best test accuracy for each label
    test_best = test_df.loc[test_df.groupby("label")["accuracy"].idxmax()]

    # Merge train statistics with test accuracies for the overall results table
    stats_df = pd.merge(
        train_stats,
        test_best[["label", "accuracy", "run"]].rename(
            columns={"accuracy": "test_accuracy", "run": "test_best_run"}
        ),
        on="label",
        how="outer",
    )

    print("\nAccuracy Statistics by Label:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(stats_df.to_string(index=False))

    # Print markdown tables
    print("\nMarkdown Tables:")
    print("\nFull Results Table:")
    print(format_markdown_table(stats_df))
    print("\nTest Metrics Table:")
    print(format_test_metrics_markdown(test_best))

    # Print LaTeX tables
    print("\nLaTeX Tables:")
    print("\nFull Results Table:")
    print(format_latex_table(stats_df))
    print("\nTest Metrics Table:")
    print(format_test_metrics_latex(test_best))


if __name__ == "__main__":
    main()
