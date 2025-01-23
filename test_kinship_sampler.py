import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets.scl import SCLDataModule


def analyze_epoch_sampling(sampler):
    """Analyze sampling patterns over one epoch."""
    individual_counts = defaultdict(int)
    relationship_type_counts = defaultdict(int)
    family_counts = defaultdict(int)
    batch_times = []

    start_time = time.time()
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
            sampler=True,
            sampling_weights=weights,
            sampler_verbose=False,
            num_workers=16,
            sampler_score_update_period=sampler_score_update_period,
            sampler_max_pairs_per_update=sampler_max_pairs_per_update,
        )
        dm.setup("fit")

        # Get the sampler
        sampler = dm.train_dataloader().batch_sampler.sampler

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


def main():
    # Number of trials per configuration
    N_TRIALS = 5

    # Define configurations to test
    configs = [
        # Baseline for comparison (no weighting, random sampling)
        {
            "weights": {"rel": 0.0, "fam": 0.0, "ind": 0.0, "diff": 0.0},
            "sampler_score_update_period": 1,
            "sampler_max_pairs_per_update": 0,  # use all pairs
            "name": "baseline",
        },
        # Baseline (pure random, limited pairs and updates)
        {
            "weights": {"rel": 0.0, "fam": 0.0, "ind": 0.0, "diff": 0.0},
            "sampler_score_update_period": 5,
            "sampler_max_pairs_per_update": 100,
            "name": "baseline_limited",
        },
        # Pure relationship weighting (full pairs, full updates)
        {
            "weights": {"rel": 1.0, "fam": 0.0, "ind": 0.0, "diff": 0.0},
            "sampler_score_update_period": 1,
            "sampler_max_pairs_per_update": 0,
            "name": "pure_relationship",
        },
        # Pure relationship weighting (limited pairs, full updates)
        {
            "weights": {"rel": 1.0, "fam": 0.0, "ind": 0.0, "diff": 0.0},
            "sampler_score_update_period": 5,
            "sampler_max_pairs_per_update": 100,
            "name": "pure_relationship_limited",
        },
        # Pure difficulty weighting (full pairs, full updates)
        {
            "weights": {"rel": 0.0, "fam": 0.0, "ind": 0.0, "diff": 1.0},
            "sampler_score_update_period": 1,
            "sampler_max_pairs_per_update": 0,
            "name": "pure_difficulty",
        },
        # Pure difficulty weighting (limited pairs, full updates)
        {
            "weights": {"rel": 0.0, "fam": 0.0, "ind": 0.0, "diff": 1.0},
            "sampler_score_update_period": 5,
            "sampler_max_pairs_per_update": 100,
            "name": "pure_difficulty_limited",
        },
        # Balanced weighting (full pairs, full updates)
        {
            "weights": {"rel": 0.33, "fam": 0.33, "ind": 0.34, "diff": 0.0},
            "sampler_score_update_period": 1,
            "sampler_max_pairs_per_update": 0,
            "name": "balanced",
        },
        # Balanced weighting (limited pairs, full updates)
        {
            "weights": {"rel": 0.33, "fam": 0.33, "ind": 0.34, "diff": 0.0},
            "sampler_score_update_period": 5,
            "sampler_max_pairs_per_update": 100,
            "name": "balanced_limited",
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
            n_trials=N_TRIALS,
        )

    # Create summary DataFrame
    summary = pd.DataFrame(results).T

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

    # Print the table
    print("\nSampling Results (mean ± std across trials):")
    print("CV values in %, Batch Time in ms")
    print("=" * 100)
    print(results_table.to_string())
    print("=" * 100)

    # Save detailed results
    summary.to_csv("sampler_evaluation_results.csv")
    print("\nDetailed results saved to sampler_evaluation_results.csv")


if __name__ == "__main__":
    main()
