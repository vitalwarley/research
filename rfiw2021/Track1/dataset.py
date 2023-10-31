import os

import numpy as np
import torch
from keras.preprocessing import image
from torch.utils.data import Dataset
from Track1.utils import np2tensor

FILE = os.path.dirname(os.path.abspath(__file__))


class FIW(Dataset):
    def __init__(self, sample_path, classification=False, classes=(), transform=None):
        self.sample_path = sample_path
        self.transform = transform
        self.classification = classification
        self.bias = 0
        self.name2id = {
            "non-kin": 0,
            "md": 1,
            "ms": 2,
            "sibs": 3,
            "ss": 4,
            "bb": 5,
            "fd": 6,
            "fs": 7,
            "gfgd": 8,
            "gfgs": 9,
            "gmgd": 10,
            "gmgs": 11,
        }
        # keep ids from 0 to len(classes)
        self.name2id = {k: idx for idx, k in enumerate(classes) if k in self.name2id.keys()}
        self.sample_list = self.load_sample()
        print(
            f"Loaded {len(self.sample_list)} samples from {sample_path}"
            f" with {len(self.name2id)} classes: {list(self.name2id.items())}"
        )

    def load_sample(self):
        sample_list = []
        f = open(self.sample_path, "r+", encoding="utf-8")
        while True:
            line = f.readline().replace("\n", "")
            if not line:
                break
            else:
                tmp = line.split(" ")
                if self.classification:
                    # Only add if tmp[-2] is in classes
                    if tmp[-2] in self.name2id.keys():
                        sample_list.append(tmp)
                else:
                    sample_list.append([tmp[0], tmp[1], tmp[2], tmp[-1]])
        f.close()
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        img = image.load_img(f"{FILE}/{path}", target_size=(112, 112))
        return img

    def set_bias(self, bias):
        self.bias = bias

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        # id, f1, f2, kin_relation, is_kin
        sample = self.sample_list[item + self.bias]
        img1, img2 = self.read_image(sample[1]), self.read_image(sample[2])
        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)
        img1, img2 = np2tensor(self.preprocess(np.array(img1, dtype=float))), np2tensor(
            self.preprocess(np.array(img2, dtype=float))
        )
        if self.classification:
            label = self.name2id[sample[-2]]  # kin relation
            if not int(sample[-1]):
                label = 0
        else:
            label = np2tensor(np.array(sample[-1], dtype=float))
        return img1, img2, label
