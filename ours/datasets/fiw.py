from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from .utils import Sample


class FIW(Dataset):
    def __init__(self, root_dir, sample_path, transform=None):
        self.root_dir = Path(root_dir)
        self.sample_path = sample_path
        self.transform = transform
        self.bias = 0
        self.sample_list = self.load_sample()
        print(f"Loaded {len(self.sample_list)} samples from {sample_path}")

    def load_sample(self):
        sample_list = []
        lines = Path(self.root_dir, self.sample_path).read_text().strip().split("\n")
        for line in lines:
            if len(line) < 1:
                continue
            tmp = line.split(" ")
            # sample = Sample(tmp[0], tmp[1], tmp[2], tmp[-2], tmp[-1])
            # facornet
            # id, f1, f2, kin, is_kin, sim -> train
            # id, f1, f2, kin, is_kin -> val
            sample = Sample(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4])
            sample_list.append(sample)
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        # TODO: add to utils.py
        img = cv2.imread(f"{self.root_dir}/{path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        return img

    def set_bias(self, bias):
        self.bias = bias

    def __getitem__(self, item):
        # id, f1, f2, kin_relation, is_kin
        sample = self.sample_list[item + self.bias]
        img1, img2 = self.read_image(sample.f1), self.read_image(sample.f2)
        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)
        is_kin = torch.tensor(int(sample.is_kin))
        kin_id = Sample.NAME2LABEL[sample.kin_relation] if is_kin else 0
        labels = (kin_id, is_kin)
        return img1, img2, labels
