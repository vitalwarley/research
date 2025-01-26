import subprocess
from io import StringIO
import pandas as pd
import argparse
import shlex


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
        f"scl:{operation}",
    ]

    # Add any additional guild arguments
    if guild_args:
        guild_command.extend(shlex.split(guild_args))

    # Add output formatting arguments
    guild_command.extend(
        [
            "-cc",
            # dot is needed
            "run,.label,max accuracy",
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
        df.columns = ["run", "label", "accuracy"]
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
    table = "| Configuration | Val Acc (CI) | Test Acc |\n"
    table += "|--------------|-------------|----------|\n"

    # Add rows
    for _, row in stats_df.iterrows():
        # Format train accuracy as mean ± std
        train_acc = f"{row['train_mean']:.5f} ± {row['train_std']:.5f}"
        # Format test accuracy
        test_acc = f"{row['test_accuracy']:.5f}"
        # Add row to table
        table += f"| {row['label']} | {train_acc} | {test_acc} |\n"

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
    args = parser.parse_args()

    # Get experiment data for train and test
    train_df = get_experiment_accuracies(args.guild_args, "train")
    test_df = get_experiment_accuracies(args.guild_args, "test")

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
    test_best = test_best.rename(
        columns={"accuracy": "test_accuracy", "run": "test_best_run"}
    )

    # Merge train statistics with test accuracies
    stats_df = pd.merge(
        train_stats,
        test_best[["label", "test_accuracy", "test_best_run"]],
        on="label",
        how="outer",
    )

    print("\nAccuracy Statistics by Label:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(stats_df.to_string(index=False))

    # Print markdown table
    print("\nMarkdown Table:")
    print(format_markdown_table(stats_df))


if __name__ == "__main__":
    main()
