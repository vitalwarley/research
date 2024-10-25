import os
import random
from collections import defaultdict
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets.fiw import FIW, FIWFamilyV3, FIWFamilyV4AG, FIWTask2
from datasets.utils import collate_fn_fiw_family_v3, collate_fn_fiw_family_v4

N_WORKERS = os.cpu_count() or 16


class KinshipBatchSampler:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_counters = defaultdict(int)
        self.indices = list(range(len(self.dataset)))
        self._shuffle_indices()

    def _shuffle_indices(self):
        random.shuffle(self.indices)

    def _get_image_with_min_count(self, person_images):
        min_count_image = min(person_images, key=lambda person: self.image_counters[person])
        return min_count_image

    def _replace_duplicates(self, sub_batch):
        family_counts = defaultdict(int)
        for pair in sub_batch:
            fam = pair[2][2]  # Label, Family ID
            family_counts[fam] += 1

        while any(count > 1 for count in family_counts.values()):
            # print(f"Family counts: {family_counts}")
            for i in range(len(sub_batch)):
                current_fam = sub_batch[i][2][2]
                if family_counts[current_fam] > 1:
                    # print(f"Checking pair {i + 1}: {current_fam}")
                    while (replacement_fam := random.choice(list(self.dataset.fam2rel.keys()))) in family_counts:
                        pass
                    replacement_pair_idx = random.choice(self.dataset.fam2rel[replacement_fam])
                    replacement_pair = self.dataset.relationships[replacement_pair_idx]
                    sub_batch[i] = replacement_pair
                    family_counts[current_fam] -= 1
                    family_counts[replacement_fam] += 1
                    # print(f"Replaced pair {i + 1} with a new pair")
        return sub_batch

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            sub_batch_indices = self.indices[i : i + self.batch_size]
            sub_batch = [self.dataset.relationships[idx] for idx in sub_batch_indices]
            sub_batch = self._replace_duplicates(sub_batch)
            batch = []

            for pair in sub_batch:
                imgs1, imgs2, _ = pair
                img1 = self._get_image_with_min_count(imgs1)
                img2 = self._get_image_with_min_count(imgs2)
                img1_id = self.dataset.persons2idx[img1]
                img2_id = self.dataset.persons2idx[img2]
                self.image_counters[img1] += 1
                self.image_counters[img2] += 1
                batch.append((img1_id, img2_id))

            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


# Example usage:
# Assuming dataset is an instance of a Dataset class where __getitem__ returns (img1, img2, labels)
# batch_size = 32
# sampler = KinshipBatchSampler(dataset, batch_size)
# data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)


class SCLDataModule(L.LightningDataModule):
    DATASETS = {"facornet": FIW, "ff-v4-ag": FIWFamilyV4AG, "ff-v3": FIWFamilyV3}
    COLLATE_FN = {"facornet": None, "ff-v4-ag": collate_fn_fiw_family_v4, "ff-v3": collate_fn_fiw_family_v3}

    def __init__(
        self,
        batch_size=20,
        root_dir=".",
        augmentation_params={},
        augment=False,
        sampler=True,
        dataset="ff-v4-ag",
        bias=False,
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
        print(f"Setup {stage} datasets")

    def train_dataloader(self):
        if self.sampler:
            sampler = KinshipBatchSampler(self.train_dataset, self.batch_size)
            batch_size = 1  # Sampler returns batches
            shuffle = False  # Cannot shuffle with sampler
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
        print(f"Setup {stage} datasets")

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
        dataset="fiw",
        batch_size=batch_size,
        root_dir="../datasets/facornet",
    )
    dm.setup("validate")
    data_loader = dm.val_dataloader()

    # Iterate over DataLoader
    for i, batch in enumerate(data_loader):
        print(f"Batch {i + 1}: ", batch[-1])
