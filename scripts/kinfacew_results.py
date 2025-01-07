import subprocess
from pathlib import Path
from typing import Tuple

import pandas as pd

# Constants
OUTPUT_DIR = Path("results") / "kinface"
OUTPUT_CSV = OUTPUT_DIR / "kinfacew_results.csv"
MEAN_METRICS_CSV = OUTPUT_DIR / "mean_metrics_by_dataset.csv"
GUILD_EXPERIMENT = "scl:kinface-ft"

# Column configurations
METRICS_COLUMNS = [
    "run",
    "label",
    "=data.init_args.dataset as dataset",
    "=data.init_args.fold as fold",
    "=data.init_args.batch_size as batch_size",
    "=model.init_args.lr as lr",
    "=model.init_args.loss.init_args.tau as tau",
    "=model.init_args.loss.init_args.alpha_neg as alpha",
    "accuracy",
    "auc",
    "accuracy/non-kin",
    "accuracy/md",
    "accuracy/ms",
    "accuracy/sibs",
]

# Mapping for kinship types correction
KINSHIP_MAPPING = {"accuracy/non-kin": "fd", "accuracy/md": "fs", "accuracy/ms": "md", "accuracy/sibs": "ms"}


def generate_guild_results() -> None:
    """Generate results CSV using Guild AI command."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    guild_command = [
        "guild",
        "compare",
        "-Fo",
        GUILD_EXPERIMENT,
        "-cc",
        ",".join(METRICS_COLUMNS),
        "--csv",
        str(OUTPUT_CSV),  # Convert Path to string for subprocess
    ]

    try:
        subprocess.run(guild_command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute Guild command: {e}")


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the results data.

    Returns:
        tuple: (main_results_df, cross_eval_df)
    """
    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(f"Failed to generate {OUTPUT_CSV}")

    df = pd.read_csv(OUTPUT_CSV)

    # Rename columns and reorder
    df = df.rename(columns=KINSHIP_MAPPING)
    columns_order = [
        "run",
        "label",
        "batch_size",
        "dataset",
        "fold",
        "lr",
        "alpha",
        "tau",
        "accuracy",
        "auc",
        "fd",
        "fs",
        "md",
        "ms",
    ]
    df = df[columns_order]

    # Split main results and cross-evaluation
    return df.iloc[:-2, :], df.iloc[-2:, :]


def calculate_mean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean metrics grouped by dataset, batch_size, and learning rate.

    Args:
        df: Input DataFrame with all results

    Returns:
        DataFrame with mean metrics
    """
    metrics = ["accuracy", "auc", "fd", "fs", "md", "ms"]
    return df.groupby(["dataset", "batch_size", "lr"]).agg({metric: "mean" for metric in metrics}).reset_index()


def main():
    """Main execution function."""
    # Generate results using Guild
    generate_guild_results()

    # Load and preprocess data
    main_results, cross_eval = load_and_preprocess_data()

    # Calculate mean metrics
    mean_results = calculate_mean_metrics(main_results)

    # Print results
    print("Mean Accuracy and AUC by Dataset:")
    print(mean_results.to_string(index=False))

    # Save results
    mean_results.to_csv(MEAN_METRICS_CSV, index=False)
    print("\nResults saved to:")
    print(f"- {OUTPUT_CSV}")
    print(f"- {MEAN_METRICS_CSV}")


if __name__ == "__main__":
    main()
