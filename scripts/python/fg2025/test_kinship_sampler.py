import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.scl import SCLDataModule

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def analyze_epoch_sampling(sampler):
    """Analyze sampling patterns over one epoch."""
    individual_counts = defaultdict(int)
    relationship_type_counts = defaultdict(int)
    family_counts = defaultdict(int)
    batch_times = []

    start_time = time.time()

    # Handle FIW dataset (no sampler) case
    if isinstance(sampler, DataLoader) and not hasattr(
        sampler.batch_sampler.sampler, "_get_person_ids"
    ):
        # Get the underlying dataset and shuffle its samples
        dataset = sampler.dataset
        samples = dataset.sample_list.copy()
        np.random.shuffle(samples)

        # Process each sample
        for sample in tqdm(samples, desc="Analyzing samples", leave=False):
            # Record batch time
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            start_time = time.time()

            # Update relationship type counts
            relationship_type_counts[sample.kin_relation] += 1

            # Update family counts
            family_counts[sample.f1fid] += 1

            # Update individual counts by combining family id and member id
            person1_id = f"F{sample.f1fid:04d}_MID{sample.f1mid}"
            person2_id = f"F{sample.f2fid:04d}_MID{sample.f2mid}"
            individual_counts[person1_id] += 1
            individual_counts[person2_id] += 1
    else:
        # Original sampler case
        for batch in tqdm(sampler, desc="Analyzing batches", leave=False):
            # Record batch time
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            start_time = time.time()

            for _, _, labels in batch:
                # Update relationship type counts
                rel_type = labels[4]
                relationship_type_counts[rel_type] += 1

                # Update family counts
                fam = labels[2]
                family_counts[fam] += 1

                # Update individual counts
                person1_id, person2_id = sampler._get_person_ids(labels)
                individual_counts[person1_id] += 1
                individual_counts[person2_id] += 1

    return individual_counts, relationship_type_counts, family_counts, batch_times


def calculate_cv(counts):
    """Calculate coefficient of variation."""
    values = np.array(list(counts.values()))
    return (np.std(values) / np.mean(values)) * 100 if len(values) > 0 else 0


def print_sampling_statistics(
    individual_counts, relationship_type_counts, family_counts, batch_times, trial=None
):
    """Print detailed sampling statistics."""
    trial_str = f" (Trial {trial})" if trial is not None else ""
    print(f"\nSampling Statistics{trial_str}:")
    print("-" * 50)

    # Timing statistics
    print("Timing Statistics:")
    print(f"Total batches: {len(batch_times)}")
    print(f"Mean time per batch: {np.mean(batch_times):.4f}s")
    print(f"Std time per batch: {np.std(batch_times):.4f}s")
    print(f"Min time per batch: {np.min(batch_times):.4f}s")
    print(f"Max time per batch: {np.max(batch_times):.4f}s")
    print()

    # Individual statistics
    total_samples = sum(individual_counts.values())
    unique_individuals = len(individual_counts)
    mean_samples = total_samples / unique_individuals if unique_individuals > 0 else 0
    max_samples = max(individual_counts.values()) if individual_counts else 0
    min_samples = min(individual_counts.values()) if individual_counts else 0

    print("Individual Sampling:")
    print(f"Total samples: {total_samples}")
    print(f"Unique individuals: {unique_individuals}")
    print(f"Mean samples per individual: {mean_samples:.2f}")
    print(f"Max samples: {max_samples}")
    print(f"Min samples: {min_samples}")
    print(f"Individual CV: {calculate_cv(individual_counts):.1f}%")

    # Relationship type statistics
    print("\nRelationship Types:")
    total_rels = sum(relationship_type_counts.values())
    for rel_type, count in sorted(
        relationship_type_counts.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / total_rels) * 100 if total_rels > 0 else 0
        print(f"{rel_type}: {count} ({percentage:.1f}%)")
    print(f"Relationship CV: {calculate_cv(relationship_type_counts):.1f}%")

    # Family statistics
    print("\nFamily Sampling:")
    total_families = len(family_counts)
    mean_family_samples = (
        sum(family_counts.values()) / total_families if total_families > 0 else 0
    )
    print(f"Total families: {total_families}")
    print(f"Mean samples per family: {mean_family_samples:.2f}")
    print(f"Max samples: {max(family_counts.values()) if family_counts else 0}")
    print(f"Min samples: {min(family_counts.values()) if family_counts else 0}")
    print(f"Family CV: {calculate_cv(family_counts):.1f}%")


def evaluate_weight_configuration(
    weights,
    batch_size=64,
    root_dir="data/fiw/track1",
    sampler_score_update_period=5,
    sampler_max_pairs_per_update=100,
    n_trials=5,
    verbose=True,
):
    """Evaluate a specific weight configuration with multiple trials."""
    if verbose:
        if weights is None:
            print("\nEvaluating baseline (no sampler)")
        else:
            print(f"\nEvaluating weights: {weights}")
            print(
                f"Update period: {sampler_score_update_period}, Max pairs per update: {sampler_max_pairs_per_update}"
            )

    # Store results for each trial
    trial_metrics = []

    for trial in range(n_trials):
        if verbose:
            print(f"\nTrial {trial + 1}/{n_trials}")

        # Initialize dataset with the weight configuration
        dm = SCLDataModule(
            dataset="ff-v3",
            batch_size=batch_size,
            root_dir=root_dir,
            sampler=weights is not None,  # Only use sampler if weights are provided
            sampling_weights=weights,
            sampler_verbose=False,
            num_workers=16,
            sampler_score_update_period=sampler_score_update_period,
            sampler_max_pairs_per_update=sampler_max_pairs_per_update,
        )
        dm.setup("fit")

        # Get the sampler or dataloader depending on configuration
        if weights is not None:
            sampler = dm.train_dataloader().batch_sampler.sampler
        else:
            sampler = dm.train_dataloader()

        # Analyze sampling patterns
        individual_counts, relationship_type_counts, family_counts, batch_times = (
            analyze_epoch_sampling(sampler)
        )

        # Calculate metrics for this trial
        trial_metrics.append(
            {
                "individual_cv": calculate_cv(individual_counts),
                "relationship_cv": calculate_cv(relationship_type_counts),
                "family_cv": calculate_cv(family_counts),
                "mean_batch_time": np.mean(batch_times),
                "std_batch_time": np.std(batch_times),
                "min_batch_time": np.min(batch_times),
                "max_batch_time": np.max(batch_times),
            }
        )

        if verbose:
            print_sampling_statistics(
                individual_counts,
                relationship_type_counts,
                family_counts,
                batch_times,
                trial + 1,
            )

    # Calculate mean and std of metrics across trials
    trial_df = pd.DataFrame(trial_metrics)
    metrics = {}

    for column in trial_df.columns:
        metrics[f"{column}_mean"] = trial_df[column].mean()
        metrics[f"{column}_std"] = trial_df[column].std()

    return metrics


def save_markdown_table(df, output_path):
    """Save DataFrame as a markdown table."""
    with open(output_path, "w") as f:
        f.write("# Sampling Results (mean ± std across trials)\n\n")
        f.write("CV values in %, Batch Time in ms\n\n")
        f.write(df.to_markdown())


def save_latex_table(df, output_path):
    """Save DataFrame as a LaTeX table."""
    with open(output_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Sampling Results (mean ± std across trials)}\n")
        f.write("\\small\n")
        f.write(df.to_latex(escape=False))
        f.write("\\caption*{CV values in \\%, Batch Time in ms}\n")
        f.write("\\end{table}\n")


def run_evaluations(n_trials=5):
    """Run all sampler evaluations and return results."""
    # Define configurations to test
    configs = [
        # True baseline (no sampler)
        {
            "weights": None,  # no weights when not using sampler
            "sampler_score_update_period": None,
            "sampler_max_pairs_per_update": None,
            "name": "baseline",
        },
        # Random sampling with sampler (no weighting)
        {
            "weights": {"rel": 0.0, "fam": 0.0, "ind": 0.0, "diff": 0.0},
            "sampler_score_update_period": 1,
            "sampler_max_pairs_per_update": 0,  # use all pairs
            "name": "sampler_random",
        },
        # Random sampling with sampler (limited pairs and updates)
        {
            "weights": {"rel": 0.0, "fam": 0.0, "ind": 0.0, "diff": 0.0},
            "sampler_score_update_period": 5,
            "sampler_max_pairs_per_update": 100,
            "name": "sampler_random_limited",
        },
        # Pure relationship weighting (full pairs, full updates)
        {
            "weights": {"rel": 1.0, "fam": 0.0, "ind": 0.0, "diff": 0.0},
            "sampler_score_update_period": 1,
            "sampler_max_pairs_per_update": 0,
            "name": "sampler_relationship",
        },
        # Pure relationship weighting (limited pairs, full updates)
        {
            "weights": {"rel": 1.0, "fam": 0.0, "ind": 0.0, "diff": 0.0},
            "sampler_score_update_period": 5,
            "sampler_max_pairs_per_update": 100,
            "name": "sampler_relationship_limited",
        },
        # Pure difficulty weighting (full pairs, full updates)
        {
            "weights": {"rel": 0.0, "fam": 0.0, "ind": 0.0, "diff": 1.0},
            "sampler_score_update_period": 1,
            "sampler_max_pairs_per_update": 0,
            "name": "sampler_difficulty",
        },
        # Pure difficulty weighting (limited pairs, full updates)
        {
            "weights": {"rel": 0.0, "fam": 0.0, "ind": 0.0, "diff": 1.0},
            "sampler_score_update_period": 5,
            "sampler_max_pairs_per_update": 100,
            "name": "sampler_difficulty_limited",
        },
        # Balanced weighting (full pairs, full updates)
        {
            "weights": {"rel": 0.33, "fam": 0.33, "ind": 0.34, "diff": 0.0},
            "sampler_score_update_period": 1,
            "sampler_max_pairs_per_update": 0,
            "name": "sampler_balanced",
        },
        # Balanced weighting (limited pairs, full updates)
        {
            "weights": {"rel": 0.33, "fam": 0.33, "ind": 0.34, "diff": 0.0},
            "sampler_score_update_period": 5,
            "sampler_max_pairs_per_update": 100,
            "name": "sampler_balanced_limited",
        },
    ]

    # Evaluate each configuration
    results = {}
    for config in configs:
        print(f"\nTesting {config['name']} configuration")
        results[config["name"]] = evaluate_weight_configuration(
            weights=config["weights"],
            sampler_score_update_period=config["sampler_score_update_period"],
            sampler_max_pairs_per_update=config["sampler_max_pairs_per_update"],
            n_trials=n_trials,
        )

    return pd.DataFrame(results).T


def main(output_dir="results"):
    """Main function with parametrized output directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output paths
    csv_path = os.path.join(output_dir, "sampler_evaluation_results.csv")
    md_path = os.path.join(output_dir, "sampler_evaluation_results.md")
    latex_path = os.path.join(output_dir, "sampler_evaluation_results.tex")

    # Load or generate results
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}")
        summary = pd.read_csv(csv_path, index_col=0)
    else:
        print("Running evaluations...")
        summary = run_evaluations(n_trials=5)
        # Save detailed results
        summary.to_csv(csv_path)

    # Create a more readable table
    table_data = []
    for idx, row in summary.iterrows():
        config_data = {
            "Configuration": idx,
            "Individual CV": f"{row['individual_cv_mean']:.2f} ± {row['individual_cv_std']:.2f}",
            "Relationship CV": f"{row['relationship_cv_mean']:.2f} ± {row['relationship_cv_std']:.2f}",
            "Family CV": f"{row['family_cv_mean']:.2f} ± {row['family_cv_std']:.2f}",
            "Batch Time (s)": f"{row['mean_batch_time_mean']*1000:.1f} ± {row['mean_batch_time_std']*1000:.1f}",
        }
        table_data.append(config_data)

    results_table = pd.DataFrame(table_data)
    results_table = results_table.set_index("Configuration")

    # Save in different formats
    save_markdown_table(results_table, md_path)
    save_latex_table(results_table, latex_path)

    print(f"\nResults saved to {output_dir}/:")
    print("- CSV: sampler_evaluation_results.csv")
    print("- Markdown: sampler_evaluation_results.md")
    print("- LaTeX: sampler_evaluation_results.tex")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate kinship sampler configurations"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results (default: results)",
    )

    args = parser.parse_args()
    main(output_dir=args.output_dir)
