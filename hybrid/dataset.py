import logging
import random
from itertools import combinations, starmap
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mytypes as t
import numpy as np
from torch.utils.data import Dataset


class TestPairs(Dataset):
    def __init__(self, img_root: Path, csv_path: Path, save_ptype: bool = False, transform=None):
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


class FamiliesDataset(Dataset):
    def __init__(self, families_root: Path, uniform_family: bool = False, transform=None):
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # TODO: timm needs it. how to improve?
        # img = cv2.resize(img, (224, 224))
        if self._transform is not None:
            img = self._transform(img)
        return img, family_idx, person_idx

    def __len__(self) -> int:
        return len(self.seq)


class PairDataset(Dataset):
    def __init__(
        self,
        families_dataset: FamiliesDataset,
        mining_strategy: str = "baseline",
        num_pairs: Optional[int] = 10**5,
    ):
        super(PairDataset, self).__init__()
        self.families = families_dataset
        # TODO: simplify this
        if mining_strategy == "baseline":
            self.seq = []
            with open(self.families.families_root.parent / "fitw/val_pairs.csv", "r") as f:
                for line in f:
                    line = line.strip()
                    if len(line) < 1:
                        continue
                    img1, img2, label = line.split(",")
                    self.seq.append(
                        (
                            self.families.families_root / img1,
                            self.families.families_root / img2,
                            int(label),
                        )
                    )
        elif mining_strategy == "random":
            if num_pairs > len(self.families) ** 2:
                logging.info("number of mined pairs is greater than number of all pairs")
            first_elems = random.choices(self.families.seq, k=num_pairs)
            second_elems = random.choices(self.families.seq, k=num_pairs)
            seq = zip(first_elems, second_elems)
            self.seq = [(img1, img2, int(family1 == family2)) for (img1, family1, _), (img2, family2, _) in seq]
        elif mining_strategy == "all":
            seq = combinations(self.families.seq, 2)
            self.seq = [(img1, img2, int(family1 == family2)) for (img1, family1, _), (img2, family2, _) in seq]
        elif mining_strategy == "balanced_random":
            assert num_pairs % 2 == 0
            num_positive = num_pairs // 2
            anchor_family_idx = np.random.uniform(size=(num_positive,))
            negative_family_idx = np.random.uniform(size=(num_positive,))
            anchor_family = []
            negative_family = []
            num_families = len(self.families.families)
            for cur_family_idx, negative_idx in zip(anchor_family_idx, negative_family_idx):
                family_idx = int(cur_family_idx * num_families)
                anchor_family.append(self.families.families[family_idx])
                negative_sample = self.families.families[:family_idx]
                if family_idx < num_families - 1:
                    negative_sample += self.families.families[family_idx + 1 :]
                negative_family.append(negative_sample[int(negative_idx * (num_families - 1))])
            triplets = list(starmap(self.mine_triplets, zip(anchor_family, negative_family)))
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
    def mine_triplets(anchor_family: List[Path], negative_family: List[Path]) -> Tuple[Path, Path, Path]:
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
            img1 = cv2.imread(str(img1))
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(str(img2))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        if self.families._transform is not None:
            img1 = self.families._transform(img1)
            img2 = self.families._transform(img2)
        # print shape
        return img1, img2, label

    def __len__(self) -> int:
        return len(self.seq)
