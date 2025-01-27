import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets.fiw import (  # noqa
    FIW,
    FIWFamilyV3,
    FIWFamilyV3Task2,
    FIWGallery,
    FIWProbe,
    FIWSearchRetrieval,
    FIWTask2,
    SampleGallery,
    SampleProbe,
)
from datasets.utils import (  # noqa
    collate_fn_fiw_family_v3,
    collate_fn_fiw_family_v3_task2,
    sr_collate_fn_v2,
    worker_init_fn,
)

# Add the parent directory to sys.path using pathlib (to run standalone in ubuntu)
sys.path.append(str(Path(__file__).resolve().parent.parent))


class KinshipBatchSampler:
    def __init__(
        self,
        dataset,
        batch_size,
        sampling_weights=None,
        max_attempts=100,
        max_families_to_check=50,
        verbose=False,
        sampler_score_update_period=10,
        sampler_max_pairs_per_update=100,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_counters = defaultdict(int)
        self.indices = list(range(len(self.dataset)))
        self.max_attempts = max_attempts
        self.max_families_to_check = max_families_to_check
        self.verbose = verbose
        self.score_update_period = sampler_score_update_period
        self.max_pairs_per_update = sampler_max_pairs_per_update
        self.difficulty_scores = {}  # Store difficulty scores

        # Initialize counters
        self.family_counters = defaultdict(int)
        self.relationship_counters = defaultdict(int)
        self.individual_counters = defaultdict(int)
        self.max_family_samples = self._compute_max_family_samples()
        self.relationship_targets = self._compute_relationship_targets()

        # Cache relationship mappings
        self.rel_type_to_pairs = defaultdict(list)
        for rel_idx, rel in enumerate(self.dataset.relationships):
            rel_type = rel[2][4]  # Relationship type
            fam = rel[2][2]  # Family ID
            self.rel_type_to_pairs[rel_type].append((rel_idx, fam))

        # Initialize default weights if none provided
        self.sampling_weights = (
            None
            if sampling_weights and all(v == 0 for v in sampling_weights.values())
            else sampling_weights
        )

        # Pre-compute average samples per individual
        self.avg_samples_per_individual = (
            len(self.dataset) * 2 / len(self.dataset.person2idx)
        )

        # Pre-compute initial sampling scores if using weighted sampling
        if self.sampling_weights:
            self.pair_scores = {}
            for rel_type in self.rel_type_to_pairs:
                for idx, fam in self.rel_type_to_pairs[rel_type]:
                    self.pair_scores[(idx, fam)] = self._compute_sampling_score(
                        idx, fam
                    )
        else:
            print("No sampling weights defined. Pair selection will be random.")

        self._shuffle_indices()

        # Add score tracking
        self.score_history = []

        # Track current batch's image pairs for difficulty mapping
        self.current_batch_pairs = []

        # Initialize update counter
        self._update_counter = 0

        # Add distribution tracking
        self.distribution_history = {
            "relationship": defaultdict(list),
            "family": defaultdict(list),
            "individual": defaultdict(list),
        }
        self.tracking_interval = 100  # Track every N batches

    def _compute_relationship_frequencies(self):
        """Calculate the frequency of each relationship type in the dataset.

        Returns:
            dict: Mapping of relationship type to its frequency (count/total)
        """
        rel_counts = defaultdict(int)
        total_rels = len(self.dataset.relationships)
        for _, _, labels in self.dataset.relationships:
            rel_counts[labels[4]] += 1

        rel_frequencies = {rel: count / total_rels for rel, count in rel_counts.items()}

        if self.verbose:
            print("Relationship frequencies:")
            for rel, freq in sorted(rel_frequencies.items()):
                print(f"{rel}: {freq:.3f} ({rel_counts[rel]} pairs)")

        return rel_frequencies

    def _compute_relationship_targets(self):
        total_relationships = len(self.dataset)
        rel_types = set(labels[4] for _, _, labels in self.dataset.relationships)
        return {rel: total_relationships // len(rel_types) for rel in rel_types}

    def _compute_max_family_samples(self):
        n_families = len(self.dataset.fam2rel)
        total_samples = len(self.dataset)
        return total_samples // n_families * 2

    def _replace_duplicates(self, sub_batch):  # noqa
        family_counts = defaultdict(int)
        for pair in sub_batch:
            fam = pair[2][2]  # Family ID
            family_counts[fam] += 1

        attempts = 0
        initial_duplicates = sum(count > 1 for count in family_counts.values())

        if self.verbose and initial_duplicates > 0:
            print(
                f"Found {initial_duplicates} duplicate families in batch. Attempting replacement..."
            )

        while (
            any(count > 1 for count in family_counts.values())
            and attempts < self.max_attempts
        ):
            attempts += 1
            # Get set of all families currently in the batch
            exclude_families = {pair[2][2] for pair in sub_batch}

            for i in range(len(sub_batch)):
                current_fam = sub_batch[i][2][2]
                if family_counts[current_fam] > 1:
                    # Always use balanced replacement strategy
                    replacement_pair = self._find_balanced_replacement(exclude_families)

                    if replacement_pair:
                        family_counts = self._apply_replacement(
                            sub_batch,
                            i,
                            current_fam,
                            replacement_pair,
                            family_counts,
                            "balanced",
                        )
                        # Update exclude_families with the new family
                        exclude_families.discard(current_fam)
                        exclude_families.add(replacement_pair[2][2])

        if attempts >= self.max_attempts:
            if self.verbose:
                print(
                    f"Warning: Max attempts ({self.max_attempts}) reached. Falling back to random selection"
                )
            # Fall back to random selection for remaining duplicates
            while any(count > 1 for count in family_counts.values()):
                exclude_families = {pair[2][2] for pair in sub_batch}
                for i in range(len(sub_batch)):
                    current_fam = sub_batch[i][2][2]
                    if family_counts[current_fam] > 1:
                        replacement_pair = self._random_replacement(exclude_families)
                        if replacement_pair:
                            family_counts = self._apply_replacement(
                                sub_batch,
                                i,
                                current_fam,
                                replacement_pair,
                                family_counts,
                                "random",
                            )
                            exclude_families.discard(current_fam)
                            exclude_families.add(replacement_pair[2][2])

        if self.verbose:
            final_duplicates = sum(count > 1 for count in family_counts.values())
            print(f"Replacement complete. Remaining duplicates: {final_duplicates}")

        return sub_batch

    def _apply_replacement(
        self, sub_batch, index, current_fam, replacement_pair, family_counts, strategy
    ):
        """Apply a replacement pair to the sub-batch and update family counts.

        Args:
            sub_batch: List of relationship tuples to modify
            index: Index in sub_batch to replace
            current_fam: Current family ID being replaced
            replacement_pair: New relationship tuple to insert
            family_counts: Counter tracking family frequencies
            strategy: String describing replacement strategy used

        Returns:
            dict: Updated family counts after replacement
        """
        if self.verbose:
            print(
                f"Replaced family {current_fam} (relationship type: {sub_batch[index][2][4]})"
                f" with family {replacement_pair[2][2]} (relationship type: {replacement_pair[2][4]})"
                f" using {strategy} strategy"
            )
        sub_batch[index] = replacement_pair
        family_counts[current_fam] -= 1
        family_counts[replacement_pair[2][2]] += 1
        return family_counts

    def _random_replacement(self, exclude_families):
        """Find a random replacement relationship from a different family."""
        eligible_families = self._get_eligible_families(exclude_families)
        if not eligible_families:
            return None
        replacement_fam = random.choice(eligible_families)
        rel_indices = self.dataset.fam2rel[replacement_fam]
        rel_idx = random.choice(rel_indices)
        return self.dataset.relationships[rel_idx]

    def _get_eligible_families(self, exclude_families, max_family_samples=None):
        """Get families eligible for replacement, optionally considering sample limits.

        Args:
            exclude_families: Set/list of family IDs to exclude
            max_family_samples: Optional maximum samples per family limit
        """
        if max_family_samples is None:
            eligible = [
                fam
                for fam in self.dataset.fam2rel.keys()
                if fam not in exclude_families
            ]
        else:
            eligible = [
                fam
                for fam in self.dataset.fam2rel.keys()
                if fam not in exclude_families
                and self.family_counters[fam] < max_family_samples
            ]

        if self.verbose:
            print(f"Found {len(eligible)} eligible families for replacement")

        if self.max_families_to_check and len(eligible) > self.max_families_to_check:
            if self.verbose:
                print(
                    f"Sampling {self.max_families_to_check} families from {len(eligible)} eligible"
                )
            return random.sample(eligible, self.max_families_to_check)
        return eligible

    def _get_person_ids(self, labels):
        """Get person IDs from relationship labels.

        Args:
            labels: Tuple of (f1mid, f2mid, fid, ...)

        Returns:
            Tuple of (person1_id, person2_id)
        """
        f1mid, f2mid, fid = labels[:3]
        person1_key = f"F{fid:04d}_MID{f1mid}"
        person2_key = f"F{fid:04d}_MID{f2mid}"
        return (
            self.dataset.person2idx[person1_key],
            self.dataset.person2idx[person2_key],
        )

    def _compute_difficulty_score(self, pair_idx):
        """Compute normalized difficulty score for a pair.

        Returns:
            float: Normalized difficulty score between 0 and 1
        """
        # If no sampling weights, all pairs are equally difficult
        if not self.sampling_weights.get("diff", 0):
            return 0.0  # No difficulty score

        if pair_idx not in self.difficulty_scores:
            return 0.5  # Default medium difficulty if no score available

        # Get min/max scores for normalization
        all_scores = list(self.difficulty_scores.values())
        min_score = min(all_scores)
        max_score = max(all_scores)

        if min_score == max_score:
            return 0.5

        # Normalize to [0,1] range
        return (self.difficulty_scores[pair_idx] - min_score) / (max_score - min_score)

    def _compute_sampling_score(self, pair_idx, fam):
        """Compute a sampling score for a relationship pair using weights."""
        if not self.sampling_weights:
            return 0

        rel_type = self.dataset.relationships[pair_idx][2][4]
        labels = self.dataset.relationships[pair_idx][2]
        person1_id, person2_id = self._get_person_ids(labels)

        # Normalize counts (0-1 range)
        rel_score = (
            self.relationship_counters[rel_type] / self.relationship_targets[rel_type]
        )
        fam_score = self.family_counters[fam] / self.max_family_samples
        ind_score = (
            self.individual_counters[person1_id] + self.individual_counters[person2_id]
        ) / (2 * self.avg_samples_per_individual)

        # Get difficulty score
        diff_score = self._compute_difficulty_score(pair_idx)

        # Apply weights including difficulty
        final_score = (
            rel_score * self.sampling_weights.get("rel", 0)
            + fam_score * self.sampling_weights.get("fam", 0)
            + ind_score * self.sampling_weights.get("ind", 0)
            + diff_score * self.sampling_weights.get("diff", 0)  # Add difficulty weight
        )
        return final_score

    def _find_min_count_relationship(self, families_to_check):
        """Find relationship with minimum count using pre-computed scores."""
        families_set = set(families_to_check)

        # Get all pairs from eligible families with their scores
        eligible_pairs = [
            (pair, score)
            for pair, score in self.pair_scores.items()
            if pair[1] in families_set
        ]

        if not eligible_pairs:
            return None

        # Find minimum score
        min_score = min(p[1] for p in eligible_pairs)

        # Get all pairs with minimum score
        min_pairs = [p for p in eligible_pairs if p[1] == min_score]

        # Randomly select from pairs with minimum score
        selected_pair = random.choice(min_pairs)

        return self.dataset.relationships[selected_pair[0][0]]

    def _find_balanced_replacement(self, exclude_families):
        """Find replacement considering relationship, family and individual balance using sampling scores."""
        if not self.sampling_weights:
            return self._random_replacement(exclude_families)

        eligible_families = self._get_eligible_families(
            exclude_families, self.max_family_samples
        )
        if not eligible_families:
            return None
        return self._find_min_count_relationship(eligible_families)

    def _shuffle_indices(self):
        random.shuffle(self.indices)

    def _get_image_with_min_count(self, person_images):
        min_count_image = min(
            person_images, key=lambda person: self.image_counters[person]
        )
        return min_count_image

    def _update_counters(self, labels):
        """Update relationship, family and individual counters and recompute sampling scores."""
        rel_type = labels[4]
        fam = labels[2]

        # Basic counter updates (always performed)
        self.relationship_counters[rel_type] += 1
        self.family_counters[fam] += 1
        person1_id, person2_id = self._get_person_ids(labels)
        self.individual_counters[person1_id] += 1
        self.individual_counters[person2_id] += 1

        if not self.sampling_weights:
            return

        # Check if we should update scores this iteration
        self._update_counter += 1
        if self._update_counter % self.score_update_period != 0:
            return

        if self.verbose:
            t0 = time.time()

        # Get all affected pairs (same family or relationship type)
        affected_pairs = []
        for (idx, f), current_score in self.pair_scores.items():
            if f == fam or self.dataset.relationships[idx][2][4] == rel_type:
                affected_pairs.append((idx, f, current_score))

        # Sort by score (lowest first) to prioritize updating worst scores
        affected_pairs.sort(key=lambda x: x[2])
        if self.max_pairs_per_update:
            affected_pairs = affected_pairs[: self.max_pairs_per_update - 1]

        # Update scores for selected pairs
        for idx, f, _ in affected_pairs:
            self.pair_scores[(idx, f)] = self._compute_sampling_score(idx, f)

        if self.verbose:
            print(f"Updated {len(affected_pairs)} pairs in {time.time() - t0:.4f}s")

        # Track distributions periodically
        if self._update_counter % self.tracking_interval == 0:
            total_rels = sum(self.relationship_counters.values())
            total_fams = sum(self.family_counters.values())
            total_inds = sum(self.individual_counters.values())

            # Store current distributions
            for rel, count in self.relationship_counters.items():
                self.distribution_history["relationship"][rel].append(
                    count / total_rels if total_rels else 0
                )

            for fam, count in self.family_counters.items():
                self.distribution_history["family"][fam].append(
                    count / total_fams if total_fams else 0
                )

            for ind, count in self.individual_counters.items():
                self.distribution_history["individual"][ind].append(
                    count / total_inds if total_inds else 0
                )

    def get_sampling_stats(self):
        """Compute coefficient of variation for family, relationship and individual sampling."""

        def compute_cv(counter):
            values = list(counter.values())
            if not values:
                return 0
            mean = sum(values) / len(values)
            if mean == 0:
                return 0
            std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
            return std / mean

        return {
            "family_cv": compute_cv(self.family_counters),
            "relationship_cv": compute_cv(self.relationship_counters),
            "individual_cv": compute_cv(self.individual_counters),
        }

    def get_distribution_history(self):
        """Return the sampling distribution history for analysis."""
        return self.distribution_history

    def get_score_statistics(self):
        """Return score history for analysis."""
        return self.score_history

    def update_difficulty_scores(self, item_idx, difficulty_score):
        """Update difficulty score for a specific pair.

        Args:
            difficulty_score: New difficulty score based on model predictions
        """
        if self.sampling_weights:
            self.difficulty_scores[item_idx] = difficulty_score
            # Update sampling score for this pair
            fam = self.dataset.relationships[item_idx][2][2]
            self.pair_scores[(item_idx, fam)] = self._compute_sampling_score(
                item_idx, fam
            )

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            sub_batch_indices = self.indices[i : i + self.batch_size]
            sub_batch = [self.dataset.relationships[idx] for idx in sub_batch_indices]
            sub_batch = self._replace_duplicates(sub_batch)

            batch = []
            self.current_batch_pairs = []  # Reset for new batch

            start_time = time.time()
            for pair in sub_batch:
                imgs1, imgs2, labels = pair
                img1 = self._get_image_with_min_count(imgs1)
                img2 = self._get_image_with_min_count(imgs2)

                img1_id = self.dataset.image2idx[img1]
                img2_id = self.dataset.image2idx[img2]

                # Store the relationship index for this image pair
                rel_idx = self.dataset.relationships.index((imgs1, imgs2, labels))
                self.current_batch_pairs.append(rel_idx)

                self.image_counters[img1] += 1
                self.image_counters[img2] += 1
                batch.append((img1_id, img2_id, labels))
                self._update_counters(labels)

            if self.verbose:
                print(f"Processing all pairs took: {time.time() - start_time:.4f}s")

            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


class TriSubjectBatchSampler:
    """Batch sampler for tri-subject verification (Task 2).

    Similar to KinshipBatchSampler but handles triplets (father, mother, child) instead of pairs.
    Maintains balanced sampling across families, relationship types, and individuals.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        sampling_weights=None,
        max_attempts=100,
        max_families_to_check=50,
        verbose=False,
        sampler_score_update_period=10,
        sampler_max_pairs_per_update=100,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_counters = defaultdict(int)
        self.indices = list(range(len(self.dataset)))
        self.max_attempts = max_attempts
        self.max_families_to_check = max_families_to_check
        self.verbose = verbose
        self.score_update_period = sampler_score_update_period
        self.max_pairs_per_update = sampler_max_pairs_per_update
        self.difficulty_scores = {}

        # Initialize counters
        self.family_counters = defaultdict(int)
        self.relationship_counters = defaultdict(int)
        self.individual_counters = defaultdict(int)
        self.max_family_samples = self._compute_max_family_samples()
        self.relationship_targets = self._compute_relationship_targets()

        # Cache relationship mappings
        self.rel_type_to_triplets = defaultdict(list)
        for rel_idx, rel in enumerate(self.dataset.relationships):
            rel_type = rel[3][5]  # kin_relation from labels tuple
            fam = rel[3][3]  # family id from labels tuple
            self.rel_type_to_triplets[rel_type].append((rel_idx, fam))

        # Initialize sampling weights
        self.sampling_weights = (
            None
            if sampling_weights and all(v == 0 for v in sampling_weights.values())
            else sampling_weights
        )

        # Pre-compute average samples per individual
        self.avg_samples_per_individual = (
            len(self.dataset) * 3 / len(self.dataset.person2idx)  # *3 because triplets
        )

        # Pre-compute initial sampling scores if using weighted sampling
        if self.sampling_weights:
            self.triplet_scores = {}
            for rel_type in self.rel_type_to_triplets:
                for idx, fam in self.rel_type_to_triplets[rel_type]:
                    self.triplet_scores[(idx, fam)] = self._compute_sampling_score(
                        idx, fam
                    )
        else:
            print("No sampling weights defined. Triplet selection will be random.")

        self._shuffle_indices()

        # Add score tracking
        self.score_history = []
        self.current_batch_triplets = []
        self._update_counter = 0

        # Add distribution tracking
        self.distribution_history = {
            "relationship": defaultdict(list),
            "family": defaultdict(list),
            "individual": defaultdict(list),
        }
        self.tracking_interval = 100

    def _compute_relationship_frequencies(self):
        """Calculate the frequency of each relationship type in the dataset."""
        rel_counts = defaultdict(int)
        total_rels = len(self.dataset.relationships)
        for _, _, _, labels in self.dataset.relationships:
            rel_counts[labels[5]] += 1

        rel_frequencies = {rel: count / total_rels for rel, count in rel_counts.items()}

        if self.verbose:
            print("Relationship frequencies:")
            for rel, freq in sorted(rel_frequencies.items()):
                print(f"{rel}: {freq:.3f} ({rel_counts[rel]} triplets)")

        return rel_frequencies

    def _compute_relationship_targets(self):
        total_relationships = len(self.dataset)
        rel_types = set(labels[5] for _, _, _, labels in self.dataset.relationships)
        return {rel: total_relationships // len(rel_types) for rel in rel_types}

    def _compute_max_family_samples(self):
        n_families = len(self.dataset.fam2rel)
        total_samples = len(self.dataset)
        return total_samples // n_families * 3  # *3 because triplets

    def _compute_difficulty_score(self, triplet_idx):
        """Compute normalized difficulty score for a triplet based on both parent-child relationships."""
        if not self.sampling_weights.get("diff", 0):
            return 0.0

        if triplet_idx not in self.difficulty_scores:
            return 0.5

        # Get min/max scores for normalization
        all_scores = list(self.difficulty_scores.values())
        min_score = min(all_scores)
        max_score = max(all_scores)

        if min_score == max_score:
            return 0.5

        return (self.difficulty_scores[triplet_idx] - min_score) / (
            max_score - min_score
        )

    def _compute_sampling_score(self, triplet_idx, fam):
        """Compute sampling score for a triplet using weights."""
        if not self.sampling_weights:
            return 0

        rel_type = self.dataset.relationships[triplet_idx][3][5]  # kin_relation
        labels = self.dataset.relationships[triplet_idx][3]  # labels tuple
        father_id, mother_id, child_id = self._get_person_ids(labels)

        # Normalize counts (0-1 range)
        rel_score = (
            self.relationship_counters[rel_type] / self.relationship_targets[rel_type]
        )
        fam_score = self.family_counters[fam] / self.max_family_samples
        ind_score = (
            self.individual_counters[father_id]
            + self.individual_counters[mother_id]
            + self.individual_counters[child_id]
        ) / (3 * self.avg_samples_per_individual)

        diff_score = self._compute_difficulty_score(triplet_idx)

        final_score = (
            rel_score * self.sampling_weights.get("rel", 0)
            + fam_score * self.sampling_weights.get("fam", 0)
            + ind_score * self.sampling_weights.get("ind", 0)
            + diff_score * self.sampling_weights.get("diff", 0)
        )
        return final_score

    def _get_person_ids(self, labels):
        """Get person IDs from relationship labels for father, mother, and child."""
        f1mid, f2mid, f3mid, fid = labels[
            :4
        ]  # father, mother, child MIDs and family ID
        father_key = f"F{fid:04d}_MID{f1mid}"
        mother_key = f"F{fid:04d}_MID{f2mid}"
        child_key = f"F{fid:04d}_MID{f3mid}"
        return (
            self.dataset.person2idx[father_key],
            self.dataset.person2idx[mother_key],
            self.dataset.person2idx[child_key],
        )

    def _get_image_with_min_count(self, person_images):
        """Get image with minimum usage count for a person."""
        min_count_image = min(person_images, key=lambda img: self.image_counters[img])
        return min_count_image

    def _replace_duplicates(self, sub_batch):
        """Replace duplicate families in the batch to maintain diversity."""
        family_counts = defaultdict(int)
        for triplet in sub_batch:
            fam = triplet[3][3]  # family id from labels tuple
            family_counts[fam] += 1

        attempts = 0
        initial_duplicates = sum(count > 1 for count in family_counts.values())

        if self.verbose and initial_duplicates > 0:
            print(
                f"Found {initial_duplicates} duplicate families in batch. Attempting replacement..."
            )

        while (
            any(count > 1 for count in family_counts.values())
            and attempts < self.max_attempts
        ):
            attempts += 1
            exclude_families = {triplet[3][3] for triplet in sub_batch}  # family id

            for i in range(len(sub_batch)):
                current_fam = sub_batch[i][3][3]  # family id
                if family_counts[current_fam] > 1:
                    replacement_triplet = self._find_balanced_replacement(
                        exclude_families
                    )

                    if replacement_triplet:
                        family_counts = self._apply_replacement(
                            sub_batch,
                            i,
                            current_fam,
                            replacement_triplet,
                            family_counts,
                            "balanced",
                        )
                        exclude_families.discard(current_fam)
                        exclude_families.add(replacement_triplet[3][3])  # family id

        if attempts >= self.max_attempts:
            if self.verbose:
                print(
                    f"Warning: Max attempts ({self.max_attempts}) reached. Falling back to random selection"
                )
            while any(count > 1 for count in family_counts.values()):
                exclude_families = {triplet[3][3] for triplet in sub_batch}  # family id
                for i in range(len(sub_batch)):
                    current_fam = sub_batch[i][3][3]  # family id
                    if family_counts[current_fam] > 1:
                        replacement_triplet = self._random_replacement(exclude_families)
                        if replacement_triplet:
                            family_counts = self._apply_replacement(
                                sub_batch,
                                i,
                                current_fam,
                                replacement_triplet,
                                family_counts,
                                "random",
                            )
                            exclude_families.discard(current_fam)
                            exclude_families.add(replacement_triplet[3][3])  # family id

        return sub_batch

    def _apply_replacement(
        self,
        sub_batch,
        index,
        current_fam,
        replacement_triplet,
        family_counts,
        strategy,
    ):
        """Apply replacement triplet to the batch and update family counts."""
        if self.verbose:
            print(
                f"Replaced family {current_fam} (relationship type: {sub_batch[index][3][5]})"  # Fixed indices
                f" with family {replacement_triplet[3][3]} (relationship type: {replacement_triplet[3][5]})"  # Fixed indices
                f" using {strategy} strategy"
            )
        sub_batch[index] = replacement_triplet
        family_counts[current_fam] -= 1
        family_counts[replacement_triplet[3][3]] += 1  # family id
        return family_counts

    def _random_replacement(self, exclude_families):
        """Find a random replacement triplet from a different family."""
        eligible_families = self._get_eligible_families(exclude_families)
        if not eligible_families:
            return None
        replacement_fam = random.choice(eligible_families)
        rel_indices = self.dataset.fam2rel[replacement_fam]
        rel_idx = random.choice(rel_indices)
        return self.dataset.relationships[rel_idx]

    def _get_eligible_families(self, exclude_families, max_family_samples=None):
        """Get families eligible for replacement."""
        if max_family_samples is None:
            eligible = [
                fam
                for fam in self.dataset.fam2rel.keys()
                if fam not in exclude_families
            ]
        else:
            eligible = [
                fam
                for fam in self.dataset.fam2rel.keys()
                if fam not in exclude_families
                and self.family_counters[fam] < max_family_samples
            ]

        if self.verbose:
            print(f"Found {len(eligible)} eligible families for replacement")

        if self.max_families_to_check and len(eligible) > self.max_families_to_check:
            if self.verbose:
                print(
                    f"Sampling {self.max_families_to_check} families from {len(eligible)} eligible"
                )
            return random.sample(eligible, self.max_families_to_check)
        return eligible

    def _find_min_count_relationship(self, families_to_check):
        """Find relationship with minimum count using pre-computed scores."""
        families_set = set(families_to_check)

        eligible_triplets = [
            (triplet, score)
            for triplet, score in self.triplet_scores.items()
            if triplet[1] in families_set
        ]

        if not eligible_triplets:
            return None

        min_score = min(t[1] for t in eligible_triplets)
        min_triplets = [t for t in eligible_triplets if t[1] == min_score]
        selected_triplet = random.choice(min_triplets)

        return self.dataset.relationships[selected_triplet[0][0]]

    def _find_balanced_replacement(self, exclude_families):
        """Find replacement considering relationship, family and individual balance."""
        if not self.sampling_weights:
            return self._random_replacement(exclude_families)

        eligible_families = self._get_eligible_families(
            exclude_families, self.max_family_samples
        )
        if not eligible_families:
            return None
        return self._find_min_count_relationship(eligible_families)

    def _shuffle_indices(self):
        """Shuffle the dataset indices."""
        random.shuffle(self.indices)

    def _update_counters(self, labels):
        """Update relationship, family and individual counters and recompute sampling scores."""
        rel_type = labels[5]  # kin_relation
        fam = labels[3]  # family id

        self.relationship_counters[rel_type] += 1
        self.family_counters[fam] += 1
        father_id, mother_id, child_id = self._get_person_ids(labels)
        self.individual_counters[father_id] += 1
        self.individual_counters[mother_id] += 1
        self.individual_counters[child_id] += 1

        if not self.sampling_weights:
            return

        self._update_counter += 1
        if self._update_counter % self.score_update_period != 0:
            return

        if self.verbose:
            t0 = time.time()

        affected_triplets = []
        for (idx, f), current_score in self.triplet_scores.items():
            if (
                f == fam or self.dataset.relationships[idx][3][5] == rel_type
            ):  # Fixed indices
                affected_triplets.append((idx, f, current_score))

        affected_triplets.sort(key=lambda x: x[2])
        if self.max_pairs_per_update:
            affected_triplets = affected_triplets[: self.max_pairs_per_update]

        for idx, f, _ in affected_triplets:
            self.triplet_scores[(idx, f)] = self._compute_sampling_score(idx, f)

        if self.verbose:
            print(
                f"Updated {len(affected_triplets)} triplets in {time.time() - t0:.4f}s"
            )

        if self._update_counter % self.tracking_interval == 0:
            total_rels = sum(self.relationship_counters.values())
            total_fams = sum(self.family_counters.values())
            total_inds = sum(self.individual_counters.values())

            for rel, count in self.relationship_counters.items():
                self.distribution_history["relationship"][rel].append(
                    count / total_rels if total_rels else 0
                )

            for fam, count in self.family_counters.items():
                self.distribution_history["family"][fam].append(
                    count / total_fams if total_fams else 0
                )

            for ind, count in self.individual_counters.items():
                self.distribution_history["individual"][ind].append(
                    count / total_inds if total_inds else 0
                )

    def update_difficulty_scores(self, item_idx, difficulty_score):
        """Update difficulty scores for the current batch's triplets.

        For task 2, difficulty scores should be the average of father-child and mother-child difficulties.
        """
        if self.sampling_weights:
            self.difficulty_scores[item_idx] = difficulty_score
            fam = self.dataset.relationships[item_idx][3][3]
            self.triplet_scores[(item_idx, fam)] = self._compute_sampling_score(
                item_idx, fam
            )

    def get_sampling_stats(self):
        """Compute coefficient of variation for family, relationship and individual sampling."""

        def compute_cv(counter):
            values = list(counter.values())
            if not values:
                return 0
            mean = sum(values) / len(values)
            if mean == 0:
                return 0
            std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
            return std / mean

        return {
            "family_cv": compute_cv(self.family_counters),
            "relationship_cv": compute_cv(self.relationship_counters),
            "individual_cv": compute_cv(self.individual_counters),
        }

    def get_distribution_history(self):
        """Return the sampling distribution history for analysis."""
        return self.distribution_history

    def get_score_statistics(self):
        """Return score history for analysis."""
        return self.score_history

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            sub_batch_indices = self.indices[i : i + self.batch_size]
            sub_batch = [self.dataset.relationships[idx] for idx in sub_batch_indices]
            sub_batch = self._replace_duplicates(sub_batch)

            batch = []
            self.current_batch_triplets = []

            start_time = time.time()
            for triplet in sub_batch:
                father_imgs, mother_imgs, child_imgs, labels = triplet
                father_img = self._get_image_with_min_count(father_imgs)
                mother_img = self._get_image_with_min_count(mother_imgs)
                child_img = self._get_image_with_min_count(child_imgs)

                father_id = self.dataset.image2idx[father_img]
                mother_id = self.dataset.image2idx[mother_img]
                child_id = self.dataset.image2idx[child_img]

                rel_idx = self.dataset.relationships.index(triplet)
                self.current_batch_triplets.append(rel_idx)

                self.image_counters[father_img] += 1
                self.image_counters[mother_img] += 1
                self.image_counters[child_img] += 1
                batch.append((father_id, mother_id, child_id, labels))
                self._update_counters(labels)

            if self.verbose:
                print(f"Processing all triplets took: {time.time() - start_time:.4f}s")

            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


class SCLDataModule(L.LightningDataModule):
    DATASETS = {"fiw": FIW, "ff-v3": FIWFamilyV3}
    COLLATE_FN = {"fiw": None, "ff-v3": collate_fn_fiw_family_v3}

    def __init__(
        self,
        batch_size=20,
        root_dir=".",
        num_workers=None,
        augmentation_params={},
        augment=False,
        sampler=True,
        dataset="ff-v4-ag",
        bias=False,
        sampling_weights=None,
        sampler_max_attempts=100,
        sampler_max_families=50,
        sampler_verbose=False,
        sampler_score_update_period=5,
        sampler_max_pairs_per_update=100,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.augmentation_params = augmentation_params
        self.augment = augment
        if self.augment:
            self.train_transforms = T.Compose(
                [
                    T.ToPILImage(),
                    T.ColorJitter(
                        brightness=self.augmentation_params["color_jitter"][
                            "brightness"
                        ],
                        contrast=self.augmentation_params["color_jitter"]["contrast"],
                        saturation=self.augmentation_params["color_jitter"][
                            "saturation"
                        ],
                        hue=self.augmentation_params["color_jitter"]["hue"],
                    ),
                    T.RandomGrayscale(
                        p=self.augmentation_params["random_grayscale_prob"]
                    ),
                    T.RandomHorizontalFlip(
                        p=self.augmentation_params["random_horizontal_flip_prob"]
                    ),
                    T.ToTensor(),
                ]
            )
        else:
            self.train_transforms = T.Compose([T.ToTensor()])
        self.val_transforms = T.Compose([T.ToTensor()])
        self.dataset = self.DATASETS[dataset]
        self.collate_fn = self.COLLATE_FN[dataset]
        self.sampler = sampler
        self.bias = bias
        self.sampling_weights = sampling_weights
        self.sampler_max_attempts = sampler_max_attempts
        self.sampler_max_families = sampler_max_families
        self.sampler_verbose = sampler_verbose
        self.sampler_score_update_period = sampler_score_update_period
        self.sampler_max_pairs_per_update = sampler_max_pairs_per_update
        self.train_sampler = None

    def setup(self, stage=None):
        # For Hard Contrastive Loss, we need batches to have at least 1 positive pair
        # Maybe for CLWL too?
        self.shuffle = True
        # Using seed there is no problem shuffling validation set
        # Nonetheless, txt data for validation v2 and test set are ordered with is_kin=1 first. We need to shuffle it.
        # Validaton v1 doesn't need it
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=Path(self.dataset.TRAIN_PAIRS),
                batch_size=self.batch_size,
                transform=self.train_transforms,
                biased=self.bias,
            )
            self.val_dataset = FIW(
                root_dir=self.root_dir,
                sample_path=Path(FIW.VAL_PAIRS_MODEL_SEL),
                batch_size=self.batch_size,
                transform=self.val_transforms,
            )
            self.shuffle = False
        if stage == "validate" or stage is None:
            self.val_dataset = FIW(
                root_dir=self.root_dir,
                sample_path=Path(FIW.VAL_PAIRS_THRES_SEL),
                batch_size=self.batch_size,
                transform=self.val_transforms,
                shuffle=True,  # See init
            )
        if stage == "test" or stage is None:
            self.test_dataset = FIW(
                root_dir=self.root_dir,
                sample_path=Path(FIW.TEST_PAIRS),
                batch_size=self.batch_size,
                transform=self.val_transforms,
                shuffle=True,  # See init
            )
        print(f"Setup {stage or 'all'} datasets")

    def train_dataloader(self):
        if self.sampler:
            self.train_sampler = KinshipBatchSampler(
                self.train_dataset,
                self.batch_size,
                sampling_weights=self.sampling_weights,
                max_attempts=self.sampler_max_attempts,
                max_families_to_check=self.sampler_max_families,
                verbose=self.sampler_verbose,
                sampler_score_update_period=self.sampler_score_update_period,
                sampler_max_pairs_per_update=self.sampler_max_pairs_per_update,
            )
            batch_size = 1
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = not self.bias
            batch_size = self.batch_size

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )


class SCLDataModuleTask2(L.LightningDataModule):
    DATASETS = {"fiw": FIWTask2, "ff-v3": FIWFamilyV3Task2}
    COLLATE_FN = {"fiw": None, "ff-v3": collate_fn_fiw_family_v3_task2}

    def __init__(
        self,
        batch_size=20,
        root_dir=".",
        num_workers=None,
        augmentation_params={},
        augment=False,
        sampler=True,
        dataset="ff-v3",
        sampling_weights=None,
        sampler_max_attempts=100,
        sampler_max_families=50,
        sampler_verbose=False,
        sampler_score_update_period=5,
        sampler_max_pairs_per_update=100,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.augmentation_params = augmentation_params
        self.augment = augment
        if self.augment:
            self.train_transforms = T.Compose(
                [
                    T.ToPILImage(),
                    T.ColorJitter(
                        brightness=self.augmentation_params["color_jitter"][
                            "brightness"
                        ],
                        contrast=self.augmentation_params["color_jitter"]["contrast"],
                        saturation=self.augmentation_params["color_jitter"][
                            "saturation"
                        ],
                        hue=self.augmentation_params["color_jitter"]["hue"],
                    ),
                    T.RandomGrayscale(
                        p=self.augmentation_params["random_grayscale_prob"]
                    ),
                    T.RandomHorizontalFlip(
                        p=self.augmentation_params["random_horizontal_flip_prob"]
                    ),
                    T.ToTensor(),
                ]
            )
        else:
            self.train_transforms = T.Compose([T.ToTensor()])
        self.val_transforms = T.Compose([T.ToTensor()])
        self.sampler = sampler
        self.dataset = self.DATASETS[dataset]
        self.collate_fn = self.COLLATE_FN[dataset]
        self.sampling_weights = sampling_weights
        self.sampler_max_attempts = sampler_max_attempts
        self.sampler_max_families = sampler_max_families
        self.sampler_verbose = sampler_verbose
        self.sampler_score_update_period = sampler_score_update_period
        self.sampler_max_pairs_per_update = sampler_max_pairs_per_update
        self.train_sampler = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=Path(self.dataset.TRAIN_PAIRS),
                batch_size=self.batch_size,
                transform=self.train_transforms,
                sample_cls=self.dataset.SAMPLE,
            )
            self.val_dataset = FIWTask2(
                root_dir=self.root_dir,
                sample_path=Path(FIWTask2.VAL_PAIRS),
                batch_size=self.batch_size,
                transform=self.val_transforms,
                sample_cls=FIWTask2.SAMPLE,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = FIWTask2(
                root_dir=self.root_dir,
                sample_path=Path(FIWTask2.VAL_PAIRS),
                batch_size=self.batch_size,
                transform=self.val_transforms,
                sample_cls=FIWTask2.SAMPLE,
                shuffle=True,  # See init
            )
        if stage == "test" or stage is None:
            self.test_dataset = FIWTask2(
                root_dir=self.root_dir,
                sample_path=Path(FIWTask2.TEST_PAIRS),
                batch_size=self.batch_size,
                transform=self.val_transforms,
                sample_cls=FIWTask2.SAMPLE,
                shuffle=True,  # See init
            )
        print(f"Setup {stage or 'all'} datasets")

    def train_dataloader(self):
        if self.sampler:
            self.train_sampler = TriSubjectBatchSampler(
                self.train_dataset,
                self.batch_size,
                sampling_weights=self.sampling_weights,
                max_attempts=self.sampler_max_attempts,
                max_families_to_check=self.sampler_max_families,
                verbose=self.sampler_verbose,
                sampler_score_update_period=self.sampler_score_update_period,
                sampler_max_pairs_per_update=self.sampler_max_pairs_per_update,
            )
            batch_size = 1
            shuffle = False
        else:
            self.train_sampler = None
            batch_size = self.batch_size
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            sampler=self.train_sampler,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )


class SCLDataModuleTask3(L.LightningDataModule):
    def __init__(self, root_dir=".", batch_size=20, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform or T.Compose([T.ToTensor()])

    def setup(self, stage=None):
        if stage == "predict" or stage is None:
            self.probe_dataset = FIWProbe(
                root_dir=self.root_dir,
                sample_path="txt/probe.txt",
                sample_cls=SampleProbe,
                transform=self.transform,
            )
            self.gallery_dataset = FIWGallery(
                root_dir=self.root_dir,
                sample_path="txt/gallery.txt",
                sample_cls=SampleGallery,
                transform=self.transform,
            )
            self.search_retrieval = FIWSearchRetrieval(
                self.probe_dataset, self.gallery_dataset, self.batch_size
            )
        print(f"Setup {stage} datasets")

    def predict_dataloader(self):
        return DataLoader(
            self.search_retrieval,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=sr_collate_fn_v2,
            # Why num_workers reset the gallery_start_index?
            num_workers=4,
        )


if __name__ == "__main__":
    batch_size = 20
    dm = SCLDataModule(
        dataset="ff-v3",
        batch_size=batch_size,
        root_dir="data/fiw/track1",
    )
    dm.setup("fit")
    data_loader = dm.train_dataloader()

    # Iterate over DataLoader
    batch = next(iter(data_loader))
    # Get images1, images2, is_kin, kin_ids from the batch
    images, labels = batch
    # Print one image shape and labels
    print("Batch 1: ", images[0].shape, labels)

    # setup for validation set
    dm.setup("validate")
    data_loader = dm.val_dataloader()
    batch = next(iter(data_loader))
    images, labels = batch
    print("Batch 1: ", images[0].shape, labels)
