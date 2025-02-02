import subprocess
from io import StringIO
import pandas as pd


def get_experiment_metrics(operation: str, metrics: list) -> pd.DataFrame:
    """Extract metrics data from experiments for a specific operation.

    Args:
        operation: Either 'tri_subject_test' or 'search_retrieval'
        metrics: List of metrics to extract

    Returns:
        DataFrame with experiment data
    """
    # Create base guild command
    guild_command = [
        "guild",
        "compare",
        "-Fo",
        operation,
        "-cc",
        # dot is needed
        f"run,.label,{','.join(metrics)}",
        "--csv",
        "-",
    ]

    try:
        # Run guild command and capture output
        result = subprocess.run(
            guild_command, check=True, capture_output=True, text=True
        )
        # Create DataFrame from the CSV output
        df = pd.read_csv(StringIO(result.stdout))
        return df
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute Guild command: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to process Guild data: {e}")


def format_task2_table(df: pd.DataFrame) -> str:
    """Format results as a markdown table for Task 2.

    Args:
        df: DataFrame containing metrics data

    Returns:
        Markdown formatted table string
    """
    # Create table header
    table = "\n## Task 2 (Tri-Subject) Results\n"
    table += "| Configuration | Accuracy/MD | Accuracy/MS | Average |\n"
    table += "|--------------|------------|------------|----------|\n"

    # Add rows
    for _, row in df.iterrows():
        avg_acc = (row["accuracy/md"] + row["accuracy/ms"]) / 2
        table += f"| {row['label']} | {row['accuracy/md']:.5f} | {row['accuracy/ms']:.5f} | {avg_acc:.5f} |\n"

    return table


def format_task3_table(df: pd.DataFrame) -> str:
    """Format results as a markdown table for Task 3.

    Args:
        df: DataFrame containing metrics data

    Returns:
        Markdown formatted table string
    """
    # Create table header
    table = "\n## Task 3 (Search-Retrieval) Results\n"
    table += (
        "| Configuration | mAP/max | mAP/mean | Rank@5/max | Rank@5/mean | Average |\n"
    )
    table += (
        "|--------------|---------|-----------|------------|-------------|----------|\n"
    )

    # Add rows
    for _, row in df.iterrows():
        # Compute average between mAP and Rank@5 for each fusion strategy
        map_avg = (row["mAP/max"] + row["mAP/mean"]) / 2
        rank_avg = (row["rank@5/max"] + row["rank@5/mean"]) / 2
        final_avg = (map_avg + rank_avg) / 2

        table += (
            f"| {row['label']} | {row['mAP/max']:.5f} | {row['mAP/mean']:.5f} | "
            f"{row['rank@5/max']:.5f} | {row['rank@5/mean']:.5f} | {final_avg:.5f} |\n"
        )

    return table


def format_task2_latex_table(df: pd.DataFrame) -> str:
    """Format results as a LaTeX table for Task 2.

    Args:
        df: DataFrame containing metrics data

    Returns:
        LaTeX formatted table string
    """
    # Create table header
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{lccc}\n"
    table += "\\toprule\n"
    table += "Method & FMD & FMS & Avg. \\\\\n"
    table += "\\midrule\n"

    # Add rows
    for _, row in df.iterrows():
        avg_acc = (row["accuracy/md"] + row["accuracy/ms"]) / 2
        escaped_label = row["label"].replace("_", "\\_")
        table += (
            f"{escaped_label} & {row['accuracy/md']:.3f} & "
            f"{row['accuracy/ms']:.3f} & {avg_acc:.3f} \\\\\n"
        )

    # Add table footer
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Task 2 (Tri-Subject) Results}\n"
    table += "\\label{tab:task2-results}\n"
    table += "\\end{table}"

    return table


def format_task3_latex_table(df: pd.DataFrame) -> str:
    """Format results as a LaTeX table for Task 3.

    Args:
        df: DataFrame containing metrics data

    Returns:
        LaTeX formatted table string
    """
    # Create table header
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{lccc}\n"
    table += "\\toprule\n"
    table += "Method & mAP & Rank@5 & Avg. \\\\\n"
    table += "\\midrule\n"

    # Add rows
    for _, row in df.iterrows():
        base_label = row["label"].replace("_", "\\_")

        # Add row for max fusion
        max_avg = (row["mAP/max"] + row["rank@5/max"]) / 2
        table += (
            f"{base_label} (max) & {row['mAP/max']:.3f} & "
            f"{row['rank@5/max']:.3f} & {max_avg:.3f} \\\\\n"
        )

        # Add row for mean fusion
        mean_avg = (row["mAP/mean"] + row["rank@5/mean"]) / 2
        table += (
            f"{base_label} (mean) & {row['mAP/mean']:.3f} & "
            f"{row['rank@5/mean']:.3f} & {mean_avg:.3f} \\\\\n"
        )

    # Add table footer
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Task 3 (Search-Retrieval) Results}\n"
    table += "\\label{tab:task3-results}\n"
    table += "\\end{table}"

    return table


def main():
    # Get experiment data for task 2
    task2_metrics = ["accuracy/md", "accuracy/ms"]
    task2_df = get_experiment_metrics("tri_subject_test", task2_metrics)

    # Get experiment data for task 3
    task3_metrics = ["mAP/max", "mAP/mean", "rank@5/max", "rank@5/mean"]
    task3_df = get_experiment_metrics("search_retrieval", task3_metrics)

    # Print markdown tables
    print("\nMarkdown Tables:")
    print(format_task2_table(task2_df))
    print(format_task3_table(task3_df))

    # Print LaTeX tables
    print("\nLaTeX Tables:")
    print(format_task2_latex_table(task2_df))
    print("\n")
    print(format_task3_latex_table(task3_df))


if __name__ == "__main__":
    main()
