import subprocess
from io import StringIO
import pandas as pd
import argparse


def get_experiment_accuracies(n_experiments: int = 100) -> pd.DataFrame:
    """Extract accuracy data from the last N SCL train experiments."""
    # Create guild command to extract only run and accuracy
    guild_command = [
        "guild",
        "compare",
        "-Fo",
        "scl:train",
        "-n",
        str(n_experiments),
        "-cc",
        "run,max accuracy",
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


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute accuracy statistics."""
    stats = {
        "mean": df["accuracy"].mean(),
        "std": df["accuracy"].std(),
        "min": df["accuracy"].min(),
        "max": df["accuracy"].max(),
        "count": len(df),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy statistics from Guild experiments"
    )
    parser.add_argument(
        "-n",
        "--num-experiments",
        type=int,
        default=100,
        help="Number of most recent experiments to analyze",
    )
    args = parser.parse_args()

    # Get experiment data
    df = get_experiment_accuracies(args.num_experiments)

    # Compute and display statistics
    stats = compute_statistics(df)

    print("\nAccuracy Statistics:")
    print(f"Number of experiments: {stats['count']}")
    print(f"Mean accuracy: {stats['mean']:.4f}")
    print(f"Standard deviation: {stats['std']:.4f}")
    print(f"Min accuracy: {stats['min']:.4f}")
    print(f"Max accuracy: {stats['max']:.4f}")


if __name__ == "__main__":
    main()
