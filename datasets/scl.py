import os
import random
import sys
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

        # Initialize counters only if needed
        if balance_relationships:
            self.relationship_counters = defaultdict(int)
            self.relationship_targets = self._compute_relationship_targets()
            # Cache relationship mappings if needed
            self.rel_type_to_pairs = defaultdict(list)
            for rel_idx, rel in enumerate(self.dataset.relationships):
                rel_type = rel[2][4]  # Relationship type
                fam = rel[2][2]  # Family ID
                self.rel_type_to_pairs[rel_type].append((rel_idx, fam))

            # New sampling parameters
            self.MAX_PAIRS_PER_REL = 100  # Max pairs to consider per relationship

            # Pre-compute initial sampling scores
            self.pair_scores = {}
            for rel_type in self.rel_type_to_pairs:
                for idx, fam in self.rel_type_to_pairs[rel_type]:
                    self.pair_scores[(idx, fam)] = self._compute_sampling_score(idx, fam)

        self._shuffle_indices()

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
            for i in range(len(sub_batch)):
                current_fam = sub_batch[i][2][2]
                if family_counts[current_fam] > 1:
                    if self.balance_families and self.balance_relationships:
                        replacement_pair = self._find_balanced_replacement(current_fam)
                        strategy = "balanced"
                    elif self.balance_families:
                        replacement_pair = self._find_family_replacement(current_fam)
                        strategy = "family"
                    elif self.balance_relationships:
                        replacement_pair = self._find_relationship_replacement(current_fam)
                        strategy = "relationship"
                    else:
                        # Original behavior
                        replacement_pair = self._fallback_replacement(current_fam)
                        strategy = "fallback"
                    if replacement_pair:
                        self._apply_replacement(sub_batch, i, current_fam, replacement_pair, family_counts, strategy)

        if attempts >= self.max_attempts:
            if self.verbose:
                print(
                    f"Warning: Max attempts ({self.max_attempts}) reached. "
                    "Falling back to random selection for remaining duplicates"
                )
            # Fall back to random selection for remaining duplicates
            for i in range(len(sub_batch)):
                current_fam = sub_batch[i][2][2]
                if family_counts[current_fam] > 1:
                    replacement_pair = self._fallback_replacement(current_fam)
                    if replacement_pair:
                        self._apply_replacement(sub_batch, i, current_fam, replacement_pair, family_counts, "fallback")

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

    def _fallback_replacement(self, current_fam):
        """Find a random replacement relationship from a different family.

        This method provides a simple fallback strategy when more sophisticated balancing
        approaches fail. It randomly selects a different family and returns a random
        relationship from that family.

        Args:
            current_fam: The family ID to exclude from replacement selection

        Returns:
            tuple | None: A randomly selected relationship tuple from a different family,
                         or None if no eligible families exist. The tuple contains
                         (image1_path, image2_path, labels).
        """
        eligible_families = self._get_eligible_families(current_fam)
        if not eligible_families:
            return None

        replacement_fam = random.choice(eligible_families)
        rel_indices = self.dataset.fam2rel[replacement_fam]  # This is a list of indices
        rel_idx = random.choice(rel_indices)  # Choose one index
        return self.dataset.relationships[rel_idx]  # Get the relationship at that index

    def _get_eligible_families(self, current_fam, max_family_samples=None):
        """Get families eligible for replacement, optionally considering sample limits."""
        if max_family_samples is None:
            eligible = [fam for fam in self.dataset.fam2rel.keys() if fam != current_fam]
        else:
            eligible = [
                fam
                for fam in self.dataset.fam2rel.keys()
                if fam != current_fam and self.family_counters[fam] < max_family_samples
            ]

        if self.verbose:
            print(f"Found {len(eligible)} eligible families for replacement")

        if self.max_families_to_check and len(eligible) > self.max_families_to_check:
            if self.verbose:
                print(f"Sampling {self.max_families_to_check} families from {len(eligible)} eligible")
            return random.sample(eligible, self.max_families_to_check)
        return eligible

    def _compute_sampling_score(self, pair_idx, fam):
        rel_type = self.dataset.relationships[pair_idx][2][4]
        img1, img2 = self.dataset.relationships[pair_idx][:2]

        # Base scores
        rel_score = self.relationship_counters[rel_type] / self.relationship_targets[rel_type]
        fam_score = self.family_counters[fam] / self.max_family_samples

        # Image score calculation
        img1_count = max(min(self.image_counters[img] for img in img1), 1)  # avoid division by zero
        img2_count = max(min(self.image_counters[img] for img in img2), 1)

        # Use the maximum count between the two images
        # Higher count = higher score = less likely to be selected
        img_score = max(img1_count, img2_count)

        # Normalize by adding 1 to avoid division by zero and keep early samples competitive
        img_score = img_score / (max(max(self.image_counters.values(), default=0), 1) + 1)

        # Dynamic weights based on relationship type and current counts
        if rel_type in ["gmgs", "gmgd", "gfgs", "gfgd"]:
            # Rare relationships: prioritize relationship balance
            weights = {
                "rel": 0.7,  # High weight for rare relationships
                "fam": 0.2,  # Lower family weight
                "img": 0.1,  # Moderate image weight
            }
        else:
            # Common relationships: prioritize image diversity
            weights = {
                "rel": 0.2,  # Lower for common relationships
                "fam": 0.3,  # Moderate family weight
                "img": 0.5,  # Higher image weight
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

        # Sample pairs if there are too many
        if len(eligible_pairs) > self.MAX_PAIRS_PER_REL:
            eligible_pairs = random.sample(eligible_pairs, self.MAX_PAIRS_PER_REL)

        # Find pair with minimum score
        min_pair = min(eligible_pairs, key=lambda x: x[1])
        return self.dataset.relationships[min_pair[0][0]]

    def _find_balanced_replacement(self, current_fam):
        # Find replacement considering both family and relationship balance
        eligible_families = self._get_eligible_families(current_fam, self.max_family_samples)

        if not eligible_families:
            return None

        return self._find_min_count_relationship(eligible_families)

    def _find_family_replacement(self, current_fam):
        eligible_families = self._get_eligible_families(current_fam)

        if not eligible_families:
            return None

        replacement_fam = random.choice(eligible_families)
        return random.choice([self.dataset.relationships[idx] for idx in self.dataset.fam2rel[replacement_fam]])

    def _find_relationship_replacement(self, current_fam):
        # Check all families except current one, without family count restrictions
        eligible_families = self._get_eligible_families(current_fam)
        return self._find_min_count_relationship(eligible_families)

    def _shuffle_indices(self):
        random.shuffle(self.indices)

    def _get_image_with_min_count(self, person_images):
        min_count_image = min(person_images, key=lambda person: self.image_counters[person])
        return min_count_image

    def _update_counters(self, labels):
        """Update counters and sampling scores."""
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
        if self.balance_families:
            self.family_counters[fam] += 1

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            sub_batch_indices = self.indices[i : i + self.batch_size]
            sub_batch = [self.dataset.relationships[idx] for idx in sub_batch_indices]
            sub_batch = self._replace_duplicates(sub_batch)
            batch = []

            for pair in sub_batch:
                imgs1, imgs2, labels = pair
                img1 = self._get_image_with_min_count(imgs1)
                img2 = self._get_image_with_min_count(imgs2)
                img1_id = self.dataset.person2idx[img1]
                img2_id = self.dataset.person2idx[img2]
                self.image_counters[img1] += 1
                self.image_counters[img2] += 1
                batch.append((img1_id, img2_id, labels))
                self._update_counters(labels)

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
