import subprocess
from io import StringIO
import pandas as pd
import argparse
import shlex


def get_experiment_accuracies(guild_args: str) -> pd.DataFrame:
    """Extract accuracy data from experiments using provided guild arguments.
    
    Args:
        guild_args: String containing guild arguments for filtering experiments
    """
    # Create base guild command
    guild_command = [
        "guild",
        "compare",
        "-Fo",
        "scl:train",
    ]
    
    # Add any additional guild arguments
    if guild_args:
        guild_command.extend(shlex.split(guild_args))
        
    # Add output formatting arguments
    guild_command.extend([
        "-cc",
        "run,max accuracy",
        "--csv",
        "-",
    ])

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
    best_idx = df["accuracy"].idxmax()
    stats = {
        "mean": df["accuracy"].mean(),
        "std": df["accuracy"].std(),
        "min": df["accuracy"].min(),
        "max": df["accuracy"].max(),
        "count": len(df),
        "best_run": df.loc[best_idx, "run"]
    }
    return stats


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

    # Get experiment data
    df = get_experiment_accuracies(args.guild_args)

    # Print the dataframe
    print(df)

    # Compute and display statistics
    stats = compute_statistics(df)

    print("\nAccuracy Statistics:")
    print(f"Number of experiments: {stats['count']}")
    print(f"Mean accuracy: {stats['mean']:.4f}")
    print(f"Standard deviation: {stats['std']:.4f}")
    print(f"Min accuracy: {stats['min']:.4f}")
    print(f"Max accuracy: {stats['max']:.4f}")
    print(f"Best run ID: {stats['best_run']}")


if __name__ == "__main__":
    main()
