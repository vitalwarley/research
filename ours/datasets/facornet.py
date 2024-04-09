from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .fiw import FIW, FIWFamily


class FIWFaCoRNet(FIW):

    TRAIN_PAIRS = "txt/train_sort_A2_m.txt"
    VAL_PAIRS_MODEL_SEL = "txt/val_choose_A.txt"
    VAL_PAIRS_THRES_SEL = "txt/val_A.txt"
    TEST_PAIRS = "txt/test_A.txt"

    # AdaFace uses BGR -- should I revert conversion read_image here?

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, item):
        (img1, img2, labels) = super().__getitem__(item)
        # Convert img1 and img2 to BGR - they're tensors (C, H, W)
        # img1 = img1[[2, 1, 0], :, :]
        # img2 = img2[[2, 1, 0], :, :]
        return img1, img2, labels


class FIWFaCoRNetFamily(FIWFaCoRNet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Enconde all samples f1fid and f2fid to set of unique values
        self.fid_set = set()
        for sample in self.sample_list:
            self.fid_set.add(sample.f1fid)
            self.fid_set.add(sample.f2fid)
        # Map each fid to an index
        self.fid_set = sorted(list(self.fid_set))
        self.fid2idx = {fid: idx for idx, fid in enumerate(self.fid_set)}

    def _process_labels(self, sample):
        is_kin = torch.tensor(sample.is_kin)
        kin_id = self.sample_cls.NAME2LABEL[sample.kin_relation]
        fid1, fid2 = int(sample.f1fid), int(sample.f2fid)
        # Get index for each fid
        fid1, fid2 = self.fid2idx[fid1], self.fid2idx[fid2]
        labels = (kin_id, is_kin, (fid1, fid2))
        return labels


class FaCoRNetDataModule(L.LightningDataModule):
    DATASETS = {"facornet": FIWFaCoRNet, "facornet-family": FIWFaCoRNetFamily}

    def __init__(self, dataset: str, batch_size=20, root_dir=".", transform=None):
        super().__init__()
        self.dataset = self.DATASETS[
            dataset
        ]  # Couldn't find a better way to do this; if I put into init, I can't make it work
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.transform = transform or T.Compose([T.ToTensor()])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.TRAIN_PAIRS),
                batch_size=self.batch_size,
                biased=True,
                transform=self.transform,
            )
            self.val_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.VAL_PAIRS_MODEL_SEL),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.VAL_PAIRS_THRES_SEL),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.TEST_PAIRS),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        print(f"Setup {stage} datasets")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)


class FaCoRNetFamilyDataModule(FaCoRNetDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage=None):
        super().setup(stage)
        if stage == "fit" or stage is None:
            self.family_dataset = FIWFamily(
                root_dir=Path(self.root_dir, "images/Train_A/train-faces/"),
                uniform_family=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        family_dataloader = DataLoader(
            self.family_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        return [super().train_dataloader(), family_dataloader]


class FamilyDataModule(FaCoRNetDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage=None):
        super().setup(stage)
        if stage == "fit" or stage is None:
            self.family_dataset = FIWFamily(
                root_dir=Path(self.root_dir, "images/Train_A/train-faces/"),
                uniform_family=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(self.family_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
