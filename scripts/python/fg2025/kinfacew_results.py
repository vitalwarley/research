import subprocess
from pathlib import Path

import pandas as pd
import scripts.python.fg2025.utils as utils

# Constants
OUTPUT_DIR = Path("results") / "kinface"
OUTPUT_CSV = OUTPUT_DIR / "kinfacew_results.csv"
MEAN_METRICS_CSV = OUTPUT_DIR / "mean_metrics_by_dataset.csv"
DEFAULT_GUILD_EXPERIMENT = "scl:kinface-ft"

# Add new constant for TensorBoard URL
TENSORBOARD_URL = "http://localhost:12345"  # Adjust this to your TensorBoard URL

# Add new constant for default pivot metric
DEFAULT_PIVOT_METRIC = "accuracy"

# Column configurations
METADATA_COLUMNS = {
    "scl:kinface-ft": [
        "run",
        ".label",
        "=data.init_args.dataset as dataset",
        "=data.init_args.fold as fold",
        "=data.init_args.batch_size as batch_size",
        "=model.init_args.model.init_args.model as model",
        "=model.init_args.weights as weights",
        "=model.init_args.lr as lr",
        "=model.init_args.loss.init_args.tau as tau",
        "=model.init_args.loss.init_args.alpha_neg as alpha_neg",
    ],
    "facornet:kinface-ft": [
        "run",
        ".label",
        "=data.init_args.dataset as dataset",
        "=data.init_args.fold as fold",
        "=data.init_args.batch_size as batch_size",
        "=model.init_args.model.init_args.model as model",
        "=model.init_args.weights as weights",
        "=model.init_args.lr as lr",
        "=model.init_args.loss.init_args.s as s",
    ],
}

# Mapping for kinship types correction
KINSHIP_MAPPING = {"accuracy/non-kin": "fd", "accuracy/md": "fs", "accuracy/ms": "md", "accuracy/sibs": "ms"}

# Add new constant for results organization
RESULTS_BY_LABEL_DIR = OUTPUT_DIR / "by_label"

# Add new constants for mean metrics calculations
PARAM_COLUMNS = {
    "scl:kinface-ft": ["label", "weights", "model", "tau", "alpha_neg"],
    "facornet:kinface-ft": ["label", "weights", "model", "s"],
}

BASE_COLUMNS = ["label", "weights", "model"]
EXPERIMENT_COLUMNS = {"scl:kinface-ft": ["tau", "alpha_neg"], "facornet:kinface-ft": ["s"]}
COMMON_COLUMNS = ["dataset", "batch_size", "lr"]


def generate_guild_metadata(guild_experiment: str) -> pd.DataFrame:
    """Generate metadata CSV using Guild AI command and return as DataFrame.

    Args:
        guild_experiment: Guild experiment to analyze
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_csv = OUTPUT_DIR / "guild_metadata.csv"

    guild_command = [
        "guild",
        "compare",
        "-Fo",
        guild_experiment,
        "-cc",
        ",".join(METADATA_COLUMNS[guild_experiment]),
        "--csv",
        str(metadata_csv),
    ]

    try:
        subprocess.run(guild_command, check=True)
        df = pd.read_csv(metadata_csv)
        # process weights column: weights/<run_id>/exp/checkpoints/<ckpt_file> to <run_id>[:8]
        df["weights"] = df["weights"].apply(lambda x: x.split("/")[1][:8] if not pd.isna(x) else "")
        # drop runs where label doesn't start with v
        df = df[df["label"].str.startswith("v")]
        return df
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute Guild command: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to process Guild metadata: {e}")


def fetch_tensorboard_results(pivot_metric: str = DEFAULT_PIVOT_METRIC) -> pd.DataFrame:
    """Fetch results from TensorBoard for all runs.

    Args:
        pivot_metric: Metric to use for finding the best epoch (default: 'accuracy')

    Returns:
        pd.DataFrame: DataFrame with metrics for the epoch with maximum metric value
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

    # Process the data to get values at max metric epoch for each run
    results = []
    for run in df["run"].unique():
        run_data = df[df["run"] == run]

        # Find epoch with maximum specified metric
        metric_data = run_data[run_data["metric"] == pivot_metric]
        if metric_data.empty:
            raise ValueError(f"No data found for pivot metric: {pivot_metric}")

        max_metric_epoch = metric_data.loc[metric_data["Value"].idxmax(), "Step"]

        # Get all metrics for this epoch
        run_metrics = {"run": run}
        for metric in metrics_to_fetch:
            metric_at_epoch = run_data[(run_data["metric"] == metric) & (run_data["Step"] == max_metric_epoch)][
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
    merged_df = pd.merge(guild_df, tensorboard_df, on="run", how="left")

    return merged_df


def calculate_mean_metrics(df: pd.DataFrame, guild_experiment: str) -> pd.DataFrame:
    """Calculate mean metrics grouped by dataset, batch_size, and learning rate."""
    metrics = ["accuracy", "auc", "fd", "fs", "md", "ms"]
    mean_df = (
        df.groupby(["label", "dataset", "batch_size", "lr"]).agg({metric: "mean" for metric in metrics}).reset_index()
    )

    # Calculate average of kinship-specific accuracies
    mean_df["avg_kinship"] = mean_df[["fd", "fs", "md", "ms"]].mean(axis=1)

    # Get unique parameters based on experiment type
    unique_params = df[PARAM_COLUMNS[guild_experiment]].drop_duplicates()

    # Merge unique parameters with mean metrics
    mean_df = pd.merge(mean_df, unique_params, on="label", how="left")

    # Reorder columns
    mean_df = mean_df[BASE_COLUMNS + EXPERIMENT_COLUMNS[guild_experiment] + COMMON_COLUMNS + metrics + ["avg_kinship"]]

    return mean_df


def main(label: str = None, pivot_metric: str = DEFAULT_PIVOT_METRIC, guild_experiment: str = DEFAULT_GUILD_EXPERIMENT):
    """Main execution function.

    Args:
        label: Optional label to filter results. If None, processes all labels.
        pivot_metric: Metric to use for finding the best epoch (default: 'accuracy')
        guild_experiment: Guild experiment to analyze (default: 'scl:kinface-ft')
    """
    # Get metadata from Guild
    guild_metadata = generate_guild_metadata(guild_experiment)

    # Filter by label if specified
    if label:
        guild_metadata = guild_metadata[guild_metadata["label"] == label]
        if guild_metadata.empty:
            raise ValueError(f"No data found for label: {label}")

    # Get metrics from TensorBoard
    tensorboard_metrics = fetch_tensorboard_results(pivot_metric)

    # Merge the data
    main_results = merge_guild_and_tensorboard_data(guild_metadata, tensorboard_metrics)

    # Rename columns according to KINSHIP_MAPPING
    main_results = main_results.rename(columns=KINSHIP_MAPPING)

    # Calculate mean metrics
    mean_results = calculate_mean_metrics(main_results, guild_experiment)

    # Create output paths based on label
    if label:
        output_csv = RESULTS_BY_LABEL_DIR / f"kinfacew_results_{guild_experiment}_{label}.csv"
        mean_metrics_csv = RESULTS_BY_LABEL_DIR / f"mean_metrics_{guild_experiment}_{label}.csv"
    else:
        output_csv = OUTPUT_CSV
        mean_metrics_csv = MEAN_METRICS_CSV

    # Print results
    print(f"Mean Accuracy and AUC{f' for label {label}' if label else ''}:")
    print(mean_results.to_markdown(index=False))

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
    parser.add_argument(
        "--pivot-metric", type=str, default=DEFAULT_PIVOT_METRIC, help="Metric to use for finding the best epoch"
    )
    parser.add_argument(
        "--guild-experiment", type=str, default=DEFAULT_GUILD_EXPERIMENT, help="Guild experiment to analyze"
    )
    args = parser.parse_args()

    main(args.label, args.pivot_metric, args.guild_experiment)
