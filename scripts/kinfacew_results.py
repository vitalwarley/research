import subprocess
from pathlib import Path

import pandas as pd
import utils

# Constants
OUTPUT_DIR = Path("results") / "kinface"
OUTPUT_CSV = OUTPUT_DIR / "kinfacew_results.csv"
MEAN_METRICS_CSV = OUTPUT_DIR / "mean_metrics_by_dataset.csv"
GUILD_EXPERIMENT = "scl:kinface-ft"

# Add new constant for TensorBoard URL
TENSORBOARD_URL = "http://localhost:12345"  # Adjust this to your TensorBoard URL

# Column configurations
METADATA_COLUMNS = [
    "run",
    ".label",
    "=data.init_args.dataset as dataset",
    "=data.init_args.fold as fold",
    "=data.init_args.batch_size as batch_size",
    "=model.init_args.lr as lr",
    "=model.init_args.loss.init_args.tau as tau",
    "=model.init_args.loss.init_args.alpha_neg as alpha",
]

# Mapping for kinship types correction
KINSHIP_MAPPING = {"accuracy/non-kin": "fd", "accuracy/md": "fs", "accuracy/ms": "md", "accuracy/sibs": "ms"}

# Add new constant for results organization
RESULTS_BY_LABEL_DIR = OUTPUT_DIR / "by_label"


def generate_guild_metadata() -> pd.DataFrame:
    """Generate metadata CSV using Guild AI command and return as DataFrame."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_csv = OUTPUT_DIR / "guild_metadata.csv"

    guild_command = [
        "guild",
        "compare",
        "-Fo",
        GUILD_EXPERIMENT,
        "-cc",
        ",".join(METADATA_COLUMNS),
        "--csv",
        str(metadata_csv),
    ]

    try:
        subprocess.run(guild_command, check=True)
        return pd.read_csv(metadata_csv)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute Guild command: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to process Guild metadata: {e}")


def fetch_tensorboard_results() -> pd.DataFrame:
    """Fetch results from TensorBoard for all runs.

    Returns:
        pd.DataFrame: DataFrame with metrics for the epoch with maximum accuracy
    """
    # Fetch all runs
    runs = utils.fetch_runs(TENSORBOARD_URL)
    if not runs:
        raise RuntimeError("No runs found in TensorBoard")

    # Get renamed runs (optional - can be automated if needed)
    renamed_runs = {run: run for run in runs}  # Skip manual renaming

    # Define metrics to fetch (convert from METRICS_COLUMNS)
    metrics_to_fetch = ["accuracy", "auc", "accuracy/non-kin", "accuracy/md", "accuracy/ms", "accuracy/sibs"]

    # Fetch all metric data
    df = utils.load_data(TENSORBOARD_URL, metrics_to_fetch, renamed_runs)
    if df is None:
        raise RuntimeError("Failed to fetch metric data")

    # Process the data to get values at max accuracy epoch for each run
    results = []
    for run in df["run"].unique():
        run_data = df[df["run"] == run]

        # Find epoch with maximum accuracy
        accuracy_data = run_data[run_data["metric"] == "accuracy"]
        if accuracy_data.empty:
            continue

        max_accuracy_epoch = accuracy_data.loc[accuracy_data["Value"].idxmax(), "Step"]

        # Get all metrics for this epoch
        run_metrics = {"run": run}
        for metric in metrics_to_fetch:
            metric_at_epoch = run_data[(run_data["metric"] == metric) & (run_data["Step"] == max_accuracy_epoch)][
                "Value"
            ].iloc[0]
            run_metrics[metric] = metric_at_epoch

        results.append(run_metrics)

    return pd.DataFrame(results)


def merge_guild_and_tensorboard_data(guild_df: pd.DataFrame, tensorboard_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Guild metadata with TensorBoard metrics."""
    # Extract run ID from TensorBoard run path
    tensorboard_df["run"] = tensorboard_df["run"].apply(lambda x: x.split()[0])

    # Merge on run column
    merged_df = pd.merge(guild_df, tensorboard_df, on="run", how="inner")

    return merged_df


def calculate_mean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean metrics grouped by dataset, batch_size, and learning rate.

    Args:
        df: Input DataFrame with all results

    Returns:
        DataFrame with mean metrics
    """
    metrics = ["accuracy", "auc", "fd", "fs", "md", "ms"]
    mean_df = (
        df.groupby(["label", "dataset", "batch_size", "lr"]).agg({metric: "mean" for metric in metrics}).reset_index()
    )

    # Calculate average of kinship-specific accuracies
    mean_df["avg_kinship"] = mean_df[["fd", "fs", "md", "ms"]].mean(axis=1)

    return mean_df


def main(label: str = None):
    """Main execution function.

    Args:
        label: Optional label to filter results. If None, processes all labels.
    """
    # Get metadata from Guild
    guild_metadata = generate_guild_metadata()

    # Filter by label if specified
    if label:
        guild_metadata = guild_metadata[guild_metadata["label"] == label]
        if guild_metadata.empty:
            raise ValueError(f"No data found for label: {label}")

    # Get metrics from TensorBoard
    tensorboard_metrics = fetch_tensorboard_results()

    # Merge the data
    main_results = merge_guild_and_tensorboard_data(guild_metadata, tensorboard_metrics)

    # Rename columns according to KINSHIP_MAPPING
    main_results = main_results.rename(columns=KINSHIP_MAPPING)

    # Calculate mean metrics
    mean_results = calculate_mean_metrics(main_results)

    # Create output paths based on label
    if label:
        output_csv = RESULTS_BY_LABEL_DIR / f"kinfacew_results_{label}.csv"
        mean_metrics_csv = RESULTS_BY_LABEL_DIR / f"mean_metrics_{label}.csv"
    else:
        output_csv = OUTPUT_CSV
        mean_metrics_csv = MEAN_METRICS_CSV

    # Print results
    print(f"Mean Accuracy and AUC{f' for label {label}' if label else ''}:")
    print(mean_results.to_string(index=False))

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_BY_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    main_results.to_csv(output_csv, index=False)
    mean_results.to_csv(mean_metrics_csv, index=False)
    print("\nResults saved to:")
    print(f"- {output_csv}")
    print(f"- {mean_metrics_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate KinFaceW results")
    parser.add_argument("--label", type=str, help="Filter results by specific label")
    args = parser.parse_args()

    main(args.label)
