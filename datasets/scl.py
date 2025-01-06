import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms as T

# Add the parent directory to sys.path using pathlib (to run standalone in ubuntu)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets.fiw import FIW, FIWFamilyV3, FIWTask2  # noqa
from datasets.utils import collate_fn_fiw_family_v3  # noqa

N_WORKERS = os.cpu_count() or 16


class KinshipBatchSampler:
    def __init__(
        self,
        dataset,
        batch_size,
        balance_families=False,
        balance_relationships=False,
        max_attempts=100,
        max_families_to_check=50,
        verbose=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_counters = defaultdict(int)
        self.indices = list(range(len(self.dataset)))

        # Balancing flags and parameters
        self.balance_families = balance_families
        self.balance_relationships = balance_relationships
        self.max_attempts = max_attempts
        self.max_families_to_check = max_families_to_check
        self.verbose = verbose

        if self.verbose:
            print(
                f"Initializing KinshipBatchSampler with batch_size={batch_size}, "
                f"balance_families={balance_families}, balance_relationships={balance_relationships}"
            )

        self.family_counters = defaultdict(int)
        self.max_family_samples = self._compute_max_family_samples()

        self.relationship_counters = defaultdict(int)
        self.relationship_targets = self._compute_relationship_targets()
        # Cache relationship mappings if needed
        self.rel_type_to_pairs = defaultdict(list)
        for rel_idx, rel in enumerate(self.dataset.relationships):
            rel_type = rel[2][4]  # Relationship type
            fam = rel[2][2]  # Family ID
            self.rel_type_to_pairs[rel_type].append((rel_idx, fam))

        self.rel_frequencies = self._compute_relationship_frequencies()

        # Pre-compute initial sampling scores
        self.pair_scores = {}
        for rel_type in self.rel_type_to_pairs:
            for idx, fam in self.rel_type_to_pairs[rel_type]:
                self.pair_scores[(idx, fam)] = self._compute_sampling_score(idx, fam)

        self._shuffle_indices()

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
            print(f"Found {initial_duplicates} duplicate families in batch. Attempting replacement...")

        while any(count > 1 for count in family_counts.values()) and attempts < self.max_attempts:
            attempts += 1
            # Get set of all families currently in the batch
            exclude_families = {pair[2][2] for pair in sub_batch}

            for i in range(len(sub_batch)):
                current_fam = sub_batch[i][2][2]
                if family_counts[current_fam] > 1:
                    if self.balance_families and self.balance_relationships:
                        replacement_pair = self._find_balanced_replacement(exclude_families)
                        strategy = "balanced"
                    elif self.balance_families:
                        replacement_pair = self._find_family_replacement(exclude_families)
                        strategy = "family"
                    elif self.balance_relationships:
                        replacement_pair = self._find_relationship_replacement(exclude_families)
                        strategy = "relationship"
                    else:
                        replacement_pair = self._fallback_replacement(exclude_families)
                        strategy = "fallback"

                    if replacement_pair:
                        family_counts = self._apply_replacement(
                            sub_batch, i, current_fam, replacement_pair, family_counts, strategy
                        )
                        # Update exclude_families with the new family
                        exclude_families.discard(current_fam)
                        exclude_families.add(replacement_pair[2][2])

        if attempts >= self.max_attempts:
            if self.verbose:
                print(f"Warning: Max attempts ({self.max_attempts}) reached. Falling back to random selection")
            # Fall back to random selection for remaining duplicates
            while any(count > 1 for count in family_counts.values()):
                exclude_families = {pair[2][2] for pair in sub_batch}
                for i in range(len(sub_batch)):
                    current_fam = sub_batch[i][2][2]
                    if family_counts[current_fam] > 1:
                        replacement_pair = self._fallback_replacement(exclude_families)
                        if replacement_pair:
                            family_counts = self._apply_replacement(
                                sub_batch, i, current_fam, replacement_pair, family_counts, "fallback"
                            )
                            exclude_families.discard(current_fam)
                            exclude_families.add(replacement_pair[2][2])

        if self.verbose:
            final_duplicates = sum(count > 1 for count in family_counts.values())
            print(f"Replacement complete. Remaining duplicates: {final_duplicates}")

        return sub_batch

    def _apply_replacement(self, sub_batch, index, current_fam, replacement_pair, family_counts, strategy):
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

    def _fallback_replacement(self, exclude_families):
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
            eligible = [fam for fam in self.dataset.fam2rel.keys() if fam not in exclude_families]
        else:
            eligible = [
                fam
                for fam in self.dataset.fam2rel.keys()
                if fam not in exclude_families and self.family_counters[fam] < max_family_samples
            ]

        if self.verbose:
            print(f"Found {len(eligible)} eligible families for replacement")

        if self.max_families_to_check and len(eligible) > self.max_families_to_check:
            if self.verbose:
                print(f"Sampling {self.max_families_to_check} families from {len(eligible)} eligible")
            return random.sample(eligible, self.max_families_to_check)
        return eligible

    def _compute_sampling_score(self, pair_idx, fam):
        """Compute a sampling score for a relationship pair to guide selection.

        This method calculates a weighted score based on three factors:
        1. Relationship type balance - favors underrepresented relationship types
        2. Family sampling balance - prevents oversampling from single families
        3. Image usage balance - promotes diversity in image selection

        The weights are dynamically adjusted based on relationship type:
        - Rare relationships (grandparent types) prioritize relationship balance
        - Common relationships prioritize image diversity

        Args:
            pair_idx (int): Index of the relationship pair in dataset.relationships
            fam (str): Family ID for this relationship pair

        Returns:
            float: Composite score between 0-1 where lower scores indicate
                  better candidates for sampling. The score combines:
                  - Relationship type ratio (current count / target)
                  - Family sampling ratio (current count / max allowed)
                  - Image usage ratio (normalized by max usage)
        """

        rel_type = self.dataset.relationships[pair_idx][2][4]
        img1, img2 = self.dataset.relationships[pair_idx][:2]

        # Base scores (normalized between 0-1)
        if self.balance_relationships:
            rel_score = self.relationship_counters[rel_type] / self.relationship_targets[rel_type]
        else:
            rel_score = 0
        if self.balance_families:
            fam_score = self.family_counters[fam] / self.max_family_samples
        else:
            fam_score = 0

        # Image score calculation (normalized by max count)
        if self.image_counters:
            max_count = max(self.image_counters.values())
            img1_score = max((self.image_counters[img] for img in img1), default=0) / max_count
            img2_score = max((self.image_counters[img] for img in img2), default=0) / max_count
            img_score = max(img1_score, img2_score)
        else:
            img_score = 0  # When no images have been used yet

        # Calculate weights based on relationship frequency
        rel_freq = self.rel_frequencies[rel_type]
        weights = {
            "rel": 0.6 * (1 - rel_freq),  # Higher weight for rare relationships
            "fam": 0.2,  # Consistent family weight
            "img": 0.2 + (0.4 * rel_freq),  # Higher image weight for common relationships
        }

        final_score = rel_score * weights["rel"] + fam_score * weights["fam"] + img_score * weights["img"]
        return final_score

    def _find_min_count_relationship(self, families_to_check):
        """Find relationship with minimum count using pre-computed scores."""
        families_set = set(families_to_check)

        # Get all pairs from eligible families with their scores
        eligible_pairs = [(pair, score) for pair, score in self.pair_scores.items() if pair[1] in families_set]

        if not eligible_pairs:
            return None

        # Find pair with minimum score
        min_pair = min(eligible_pairs, key=lambda x: x[1])

        return self.dataset.relationships[min_pair[0][0]]

    def _find_balanced_replacement(self, exclude_families):
        # Find replacement considering both family and relationship balance
        eligible_families = self._get_eligible_families(exclude_families, self.max_family_samples)
        if not eligible_families:
            return None
        return self._find_min_count_relationship(eligible_families)

    def _find_family_replacement(self, exclude_families):
        eligible_families = self._get_eligible_families(exclude_families)
        if not eligible_families:
            return None
        replacement_fam = random.choice(eligible_families)
        return random.choice([self.dataset.relationships[idx] for idx in self.dataset.fam2rel[replacement_fam]])

    def _find_relationship_replacement(self, exclude_families):
        # Check all families except excluded ones, without family count restrictions
        eligible_families = self._get_eligible_families(exclude_families)
        return self._find_min_count_relationship(eligible_families)

    def _shuffle_indices(self):
        random.shuffle(self.indices)

    def _get_image_with_min_count(self, person_images):
        min_count_image = min(person_images, key=lambda person: self.image_counters[person])
        return min_count_image

    def _update_counters(self, labels):
        """Update relationship and family counters and recompute sampling scores.

        Updates internal counters tracking relationship type and family frequencies.
        If balancing is enabled, also recomputes sampling scores for affected pairs.
        Lower scores indicate more optimal sampling candidates, with the ideal score
        being close to 0 when both relationship type and family are undersampled.

        Args:
            labels: Tuple containing relationship metadata, where:
                labels[4] is the relationship type (e.g. 'parent-child')
                labels[2] is the family ID
        """
        rel_type = labels[4]
        fam = labels[2]
        if self.balance_relationships:
            self.relationship_counters[rel_type] += 1
            # Update sampling scores for affected relationships
            affected_pairs = [
                (idx, f)
                for idx, f in self.pair_scores.keys()
                if f == fam or self.dataset.relationships[idx][2][4] == rel_type
            ]
            for pair in affected_pairs:
                self.pair_scores[pair] = self._compute_sampling_score(pair[0], pair[1])

            # Print max and min scores
            if self.verbose:
                print(f"Max score: {max(self.pair_scores.values())}")
                print(f"Min score: {min(self.pair_scores.values())}")
        if self.balance_families:
            self.family_counters[fam] += 1

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            start_time = time.time()
            sub_batch_indices = self.indices[i : i + self.batch_size]
            if self.verbose:
                print(f"Getting batch indices took: {time.time() - start_time:.4f}s")

            start_time = time.time()
            sub_batch = [self.dataset.relationships[idx] for idx in sub_batch_indices]
            if self.verbose:
                print(f"Creating sub-batch took: {time.time() - start_time:.4f}s")

            start_time = time.time()
            sub_batch = self._replace_duplicates(sub_batch)
            if self.verbose:
                print(f"Replacing duplicates took: {time.time() - start_time:.4f}s")
            batch = []

            start_time = time.time()
            for pair in sub_batch:
                pair_start = time.time()
                imgs1, imgs2, labels = pair

                t0 = time.time()
                img1 = self._get_image_with_min_count(imgs1)
                img2 = self._get_image_with_min_count(imgs2)
                if self.verbose:
                    print(f"Getting min count images took: {time.time() - t0:.4f}s")

                t0 = time.time()
                img1_id = self.dataset.person2idx[img1]
                img2_id = self.dataset.person2idx[img2]
                if self.verbose:
                    print(f"Getting person indices took: {time.time() - t0:.4f}s")

                t0 = time.time()
                self.image_counters[img1] += 1
                self.image_counters[img2] += 1
                batch.append((img1_id, img2_id, labels))
                self._update_counters(labels)
                if self.verbose:
                    print(f"Updating counters took: {time.time() - t0:.4f}s")
                    print(f"Total pair processing took: {time.time() - pair_start:.4f}s")

            if self.verbose:
                print(f"Processing all pairs took: {time.time() - start_time:.4f}s")
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


# Example usage:
# Assuming dataset is an instance of a Dataset class where __getitem__ returns (img1, img2, labels)
# batch_size = 32
# sampler = KinshipBatchSampler(dataset, batch_size)
# data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)


class SCLDataModule(L.LightningDataModule):
    DATASETS = {"fiw": FIW, "ff-v3": FIWFamilyV3}
    COLLATE_FN = {"fiw": None, "ff-v3": collate_fn_fiw_family_v3}

    def __init__(
        self,
        batch_size=20,
        root_dir=".",
        augmentation_params={},
        augment=False,
        sampler=True,
        dataset="ff-v4-ag",
        bias=False,
        sampler_balance_families=False,
        sampler_balance_relationships=False,
        sampler_max_attempts=100,
        sampler_max_families=50,
        sampler_verbose=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.augmentation_params = augmentation_params
        self.augment = augment
        if self.augment:
            self.train_transforms = T.Compose(
                [
                    T.ToPILImage(),
                    T.ColorJitter(
                        brightness=self.augmentation_params["color_jitter"]["brightness"],
                        contrast=self.augmentation_params["color_jitter"]["contrast"],
                        saturation=self.augmentation_params["color_jitter"]["saturation"],
                        hue=self.augmentation_params["color_jitter"]["hue"],
                    ),
                    T.RandomGrayscale(p=self.augmentation_params["random_grayscale_prob"]),
                    T.RandomHorizontalFlip(p=self.augmentation_params["random_horizontal_flip_prob"]),
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
        self.sampler_balance_families = sampler_balance_families
        self.sampler_balance_relationships = sampler_balance_relationships
        self.sampler_max_attempts = sampler_max_attempts
        self.sampler_max_families = sampler_max_families
        self.sampler_verbose = sampler_verbose

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
            )
        if stage == "test" or stage is None:
            self.test_dataset = FIW(
                root_dir=self.root_dir,
                sample_path=Path(FIW.TEST_PAIRS),
                batch_size=self.batch_size,
                transform=self.val_transforms,
            )
        print(f"Setup {stage or 'all'} datasets")

    def train_dataloader(self):
        if self.sampler:
            sampler = KinshipBatchSampler(
                self.train_dataset,
                self.batch_size,
                balance_families=self.sampler_balance_families,
                balance_relationships=self.sampler_balance_relationships,
                max_attempts=self.sampler_max_attempts,
                max_families_to_check=self.sampler_max_families,
                verbose=self.sampler_verbose,
            )
            batch_size = 1
            shuffle = False
        else:
            sampler = None
            shuffle = not self.bias
            batch_size = self.batch_size

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=N_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            sampler=sampler,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=N_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=N_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )


class SCLDataModuleTask2(L.LightningDataModule):

    def __init__(
        self,
        batch_size=20,
        root_dir=".",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.train_transforms = T.Compose([T.ToTensor()])
        self.val_transforms = T.Compose([T.ToTensor()])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = FIWTask2(
                root_dir=self.root_dir,
                sample_path=Path(FIWTask2.TRAIN_PAIRS),
                batch_size=self.batch_size,
                transform=self.train_transforms,
                sample_cls=FIWTask2.SAMPLE,
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
            )
        if stage == "test" or stage is None:
            self.test_dataset = FIWTask2(
                root_dir=self.root_dir,
                sample_path=Path(FIWTask2.TEST_PAIRS),
                batch_size=self.batch_size,
                transform=self.val_transforms,
                sample_cls=FIWTask2.SAMPLE,
            )
        print(f"Setup {stage or 'all'} datasets")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=N_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=N_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=N_WORKERS,
            pin_memory=True,
            persistent_workers=True,
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
