from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from .utils import Sample


class FIW(Dataset):
    def __init__(
        self,
        root_dir: str = "",
        sample_path: str | Path = "",
        batch_size: int = 20,
        biased: bool = False,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.images_dir = "images"
        self.sample_path = Path(sample_path)
        self.batch_size = batch_size
        self.transform = transform
        self.bias = 0
        self.biased = biased
        self.sample_cls = Sample
        self.sample_list = self.load_sample()
        print(f"Loaded {len(self.sample_list)} samples from {sample_path}")

    def load_sample(self):
        print(f"Loading samples from {self.sample_path}")
        sample_list = []
        lines = Path(self.root_dir, self.sample_path).read_text().strip().split("\n")
        for line in lines:
            if len(line) < 1:
                continue
            line = line.split(" ")
            # sample = Sample(tmp[0], tmp[1], tmp[2], tmp[-2], tmp[-1])
            # facornet
            # id, f1, f2, kin, is_kin, sim -> train
            # id, f1, f2, kin, is_kin -> val
            sample = self.sample_cls(*line)
            sample_list.append(sample)
        return sample_list

    def __len__(self):
        return len(self.sample_list) // self.batch_size if self.biased else len(self.sample_list)

    def read_image(self, path):
        # TODO: add to utils.py
        img = cv2.imread(f"{self.root_dir / self.images_dir}/{path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        return img

    def set_bias(self, bias):
        if self.biased:
            self.bias = bias

    def _process_images(self, sample):
        img1, img2 = self.read_image(sample.f1), self.read_image(sample.f2)
        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2

    def _process_labels(self, sample):
        is_kin = torch.tensor(sample.is_kin)
        kin_id = self.sample_cls.NAME2LABEL[sample.kin_relation]
        # fid1, fid2 = int(sample.f1fid[1:]), int(sample.f2fid[1:])
        # labels = (kin_id, is_kin, fid1, fid2)
        labels = (kin_id, is_kin)
        # labels = is_kin
        return labels

    def __getitem__(self, item):
        # id, f1, f2, kin_relation, is_kin
        sample = self.sample_list[item + self.bias]
        (img1, img2) = self._process_images(sample)
        labels = self._process_labels(sample)
        return img1, img2, labels


class FIWFamily(Dataset):  # from rfiw2020/fitw2020/dataset.py
    def __init__(self, root_dir: Path, uniform_family: bool = False, transform=None):
        self.families_root = root_dir
        self._transform = transform
        whitelist_dir = "MID"
        self.families = [
            [
                cur_person
                for cur_person in cur_family.iterdir()
                if cur_person.is_dir() and cur_person.name.startswith(whitelist_dir)
            ]
            for cur_family in root_dir.iterdir()
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

    def __getitem__(self, idx: int):
        img_path, family_idx, person_idx = self.seq[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # TODO: timm needs it. how to improve?
        img = cv2.resize(img, (112, 112))
        if self._transform is not None:
            img = self._transform(img)
        return img, family_idx, person_idx

    def __len__(self) -> int:
        return len(self.seq)
