import logging
import random
import pickle
from itertools import combinations, starmap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from more_itertools import grouper

import mytypes as t
from typing import Tuple, Optional, Generator, Any, List, Union, Callable


class TestPairs(Dataset):
    def __init__(
        self, img_root: Path, csv_path: Path, save_ptype: bool = False, transform=None
    ):
        super(TestPairs, self).__init__()
        self.img_root = img_root
        self._transform = transform
        self.pairs = []
        with open(str(csv_path), "r") as f:
            f.readline()
            for line in f:
                idx, face1, face2, ptype = line.strip().split(",")
                self.pairs.append((self.img_root / face1, self.img_root / face2, ptype))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[t.Img, t.Img]:
        face1_path, face2_path, ptype = self.pairs[idx]
        face1 = cv2.imread(str(face1_path))
        face2 = cv2.imread(str(face2_path))
        if self._transform:
            face1 = self._transform(face1)
            face2 = self._transform(face2)
        return face1, face2, ptype


class RetrievalDataset(object):
    def __init__(
        self, gallery_root: Path, probe_root: Path, gallery_csv: Path, probe_csv: Path
    ):
        super(RetrievalDataset, self).__init__()
        self.probes = []
        self.gallery = []
        with open(str(probe_csv), "r") as f:
            f.readline()
            for line in f:
                idx, path = line.strip().split(",")
                self.probes.append((int(idx), probe_root / path))
        print("#Probes:", len(self.probes))
        with open(str(gallery_csv), "r") as f:
            f.readline()
            for line in f:
                idx, path = line.strip().split(",")
                self.gallery.append((int(idx), gallery_root / path))
        print("#Gallery:", len(self.gallery))


class FamiliesDataset(Dataset):
    def __init__(
        self, families_root: Path, uniform_family: bool = False, transform=None
    ):
        super(FamiliesDataset, self).__init__()
        self.families_root = families_root
        self._transform = transform
        whitelist_dir = "MID"
        self.families = [
            [
                cur_person
                for cur_person in cur_family.iterdir()
                if cur_person.is_dir() and cur_person.name.startswith(whitelist_dir)
            ]
            for cur_family in families_root.iterdir()
        ]
        if uniform_family:
            # TODO: implement uniform distribution of persons per family
            raise NotImplementedError
        else:
            self.seq = [
                (img_path, family_idx, person_idx)
                for family_idx, cur_family in enumerate(self.families)
                for person_idx, cur_person in enumerate(cur_family)
                for img_path in cur_person.iterdir()
            ]

    def __getitem__(self, idx: int) -> Tuple[t.Img, int, int]:
        img_path, family_idx, person_idx = self.seq[idx]
        img = cv2.imread(str(img_path))
        # TODO: timm needs it. how to improve?
        img = cv2.resize(img, (224, 224))
        if self._transform is not None:
            img = self._transform(img)
        return img, family_idx, person_idx

    def __len__(self) -> int:
        return len(self.seq)


class MS1MDataset(Dataset):
    def __init__(
        self, root: Path, transform: nn.Module = None, seq: pd.DataFrame = None
    ):
        super(MS1MDataset, self).__init__()
        self.root = root
        self._transform = transform
        self.seq = seq

    def __getitem__(self, idx: int) -> Tuple[t.Img, int, int]:
        img_path, celebrity_idx = self.seq[idx]
        img = cv2.imread(str(Path(self.root) / img_path))
        # TODO: timm needs it. how to improve?
        # img = cv2.resize(img, (112, 112))
        # img = np.moveaxis(img, 2, 0)
        if self._transform is not None:
            img = self._transform(img)
        return img, celebrity_idx

    def __len__(self) -> int:
        return len(self.seq)


class EvalPretrainDataset(Dataset):
    """LFW, CFP_FP and AGEDB_30."""

    def __init__(
        self,
        root: Path,
        target: str,
        transform: nn.Module = None,
        image_size: (int, int) = (112, 112),
    ):
        super(EvalPretrainDataset, self).__init__()
        self.root = Path(root)
        self._transform = transform
        self._image_size = image_size
        # [(data_list, issame_list)] * 3,
        #   where (data_list) -> [flip, no_flip],
        #   each one as [n_samples, 3, image_size, image_size]
        self._data = []
        self._load_dataset(target)

        print(f"{target} dataset loaded.")

    def _load_dataset(self, target):
        # read bin
        with open((self.root / target).with_suffix(".bin"), "rb") as f:
            bins, self.labels = pickle.load(f, encoding="bytes")
        # create data array
        for flip in [0, 1]:
            _first = np.empty(
                (len(self.labels), self._image_size[0], self._image_size[1], 3),
                dtype=np.float32,
            )
            _second = np.empty(
                (len(self.labels), self._image_size[0], self._image_size[1], 3),
                dtype=np.float32,
            )
            self._data.append((_first, _second))
        # populate data array
        for idx, (first, second) in enumerate(grouper(bins, 2)):
            if first is None or second is None:
                continue
            # decode _bin as image
            # TODO: is cv2.resize the same mx.image.resize_short?
            first = cv2.imdecode(first, cv2.IMREAD_COLOR)
            first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
            first = cv2.resize(first, self._image_size)
            second = cv2.imdecode(second, cv2.IMREAD_COLOR)
            second = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)
            second = cv2.resize(second, self._image_size)
            for flip in [0, 1]:
                if flip:
                    first = np.flip(first, axis=2)
                    second = np.flip(second, axis=2)
                self._data[0][flip][idx][:] = first / 255.0
                self._data[1][flip][idx][:] = second / 255.0

    def __getitem__(self, idx: int) -> Tuple[t.Img, t.Img]:
        first = (self._data[0][0][idx], self._data[0][1][idx])  # (no flip, flipped)
        second = (self._data[1][0][idx], self._data[1][1][idx])
        if self._transform is not None:
            first = (
                self._transform(self._data[0][0][idx]),
                self._transform(self._data[0][1][idx]),
            )
            second = (
                self._transform(self._data[1][0][idx]),
                self._transform(self._data[1][1][idx]),
            )

        label = self.labels[idx]
        pair = (first, second, label)
        return pair

    def __len__(self) -> int:
        return len(self.labels)


class PairDataset(Dataset):
    def __init__(
        self,
        families_dataset: FamiliesDataset,
        mining_strategy: str = "random",
        num_pairs: Optional[int] = 10**5,
    ):
        super(PairDataset, self).__init__()
        self.families = families_dataset
        if mining_strategy == "random":
            if num_pairs > len(self.families) ** 2:
                logging.info(
                    f"number of mined pairs is greater than number of all pairs"
                )
            first_elems = random.choices(self.families.seq, k=num_pairs)
            second_elems = random.choices(self.families.seq, k=num_pairs)
            seq = zip(first_elems, second_elems)
            self.seq = [
                (img1, img2, int(family1 == family2))
                for (img1, family1, _), (img2, family2, _) in seq
            ]
        elif mining_strategy == "all":
            seq = combinations(self.families.seq, 2)
            self.seq = [
                (img1, img2, int(family1 == family2))
                for (img1, family1, _), (img2, family2, _) in seq
            ]
        elif mining_strategy == "balanced_random":
            assert num_pairs % 2 == 0
            num_positive = num_pairs // 2
            anchor_family_idx = np.random.uniform(size=(num_positive,))
            negative_family_idx = np.random.uniform(size=(num_positive,))
            anchor_family = []
            negative_family = []
            num_families = len(self.families.families)
            for cur_family_idx, negative_idx in zip(
                anchor_family_idx, negative_family_idx
            ):
                family_idx = int(cur_family_idx * num_families)
                anchor_family.append(self.families.families[family_idx])
                negative_sample = self.families.families[:family_idx]
                if family_idx < num_families - 1:
                    negative_sample += self.families.families[family_idx + 1 :]
                negative_family.append(
                    negative_sample[int(negative_idx * (num_families - 1))]
                )
            triplets = list(
                starmap(self.mine_triplets, zip(anchor_family, negative_family))
            )
            positive_pairs = [(anchor, positive, 1) for anchor, positive, _ in triplets]
            negative_pairs = [(anchor, negative, 0) for anchor, _, negative in triplets]
            self.seq = positive_pairs + negative_pairs
        elif mining_strategy == "balanced_hard":
            # TODO: implement balanced with hard negative mining strategy
            raise NotImplementedError
        else:
            logging.error(f"Uknown mining strategy {mining_strategy}")
            raise NotImplementedError

    @staticmethod
    def mine_triplets(
        anchor_family: List[Path], negative_family: List[Path]
    ) -> Tuple[Path, Path, Path]:
        def random_person_img(family: List[Path]) -> Tuple[int, Path]:
            idx = np.random.randint(0, len(family))
            person = family[idx]
            all_imgs = list(person.iterdir())
            img_path = random.choice(all_imgs)
            return idx, img_path

        anchor_idx, anchor = random_person_img(anchor_family)
        _, negative = random_person_img(negative_family)
        positive_family = anchor_family[:anchor_idx]
        if anchor_idx < len(anchor_family) - 1:
            positive_family += anchor_family[anchor_idx + 1 :]
        _, positive = random_person_img(positive_family)
        return anchor, positive, negative

    def __getitem__(self, idx: int) -> Tuple[t.Img, t.Img, int]:
        img1, img2, label = self.seq[idx]
        if not isinstance(img1, np.ndarray):
            img1 = cv2.imread(str(img1)).astype(np.float32)
            img2 = cv2.imread(str(img2)).astype(np.float32)
        return img1, img2, label

    def __len__(self) -> int:
        return len(self.seq)


class ImgDataset(Dataset):
    def __init__(self, paths: List[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> t.Img:
        return cv2.imread(str(self.paths[idx])).transpose((2, 0, 1)).astype(np.float32)


class FamiliesDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        transforms: List[torch.nn.Module],
        batch_size=32,
        num_workers=8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform, self.val_transform = transforms
        self.num_workers = num_workers

        # self.dims

    def prepare_data(self):
        pass  # don't need it

    def setup(self, stage: Optional[str] = None):

        # add pretrain on MS-Celeb-1M
        if stage == "fit" or stage is None:
            # self.data_dir should be 'fitw2020'
            self.train_ds = FamiliesDataset(
                self.data_dir / "train-faces-det", transform=self.train_transform
            )
            self.val_ds = FamiliesDataset(
                self.data_dir / "val-faces-det", transform=self.val_transform
            )

        if stage == "test" or stage is None:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class MS1MDataModule(LightningDataModule):
    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        data_dir: str,
        transforms: List[torch.nn.Module] = None,
        batch_size=32,
        num_workers=8,
        val_targets="",
        debug=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.data_dir = Path(data_dir)
        self.train_save_path = str(self.data_dir / "train.npy")
        self.label_path = str(Path(self.data_dir) / "label.txt")
        self.batch_size = batch_size
        self.train_transform, self.val_transform = (
            transforms if transforms is not None else (None, None)
        )
        self.num_workers = num_workers
        self.debug = debug
        self.val_targets = val_targets

    def prepare_data(self):
        # load data
        df = pd.read_csv(self.label_path, delimiter="\t", names=["path", "target"])

        if 0 < self.num_samples < 1:
            df = df.sample(frac=self.num_samples)
        elif self.num_samples > 1:
            df = df.sample(n=self.num_samples)

        # shuffle
        df = df.sample(frac=1.0)
        classes = df.target.unique()

        if self.num_classes < len(classes):
            classes = np.random.choice(classes, size=self.num_classes, replace=False)
            df = df[df.target.isin(classes)]
            df["target"] = df.target.astype("category").cat.codes
            classes = df.target.unique()
            assert max(df.target) == max(classes), "Mismatch between classes in data"

        np.save(self.train_save_path, df.values)

    def setup(self, stage: Optional[str] = None):

        # add pretrain on MS-Celeb-1M
        if stage in (None, "fit"):
            # self.data_dir should be 'fitw2020'
            train_arr = np.load(self.train_save_path, allow_pickle=True)
            self.train_ds = MS1MDataset(
                self.data_dir, transform=self.train_transform, seq=train_arr
            )

        if stage in ("fit", "validate"):
            self.lfw = EvalPretrainDataset(
                self.data_dir, target="lfw", transform=self.val_transform
            )
        
        if stage == 'test':
            self.cfp_fp = EvalPretrainDataset(
                self.data_dir, target="cfp_fp", transform=self.val_transform
            )
            self.agedb_30 = EvalPretrainDataset(
                self.data_dir, target="agedb_30", transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        lfw_loader = DataLoader(
            self.lfw,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return lfw_loader
    
    def test_dataloader(self):
        cfp_fp_loader = DataLoader(
            self.cfp_fp,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        agedb_30_loader = DataLoader(
            self.agedb_30,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return [cfp_fp_loader, agedb_30_loader]
