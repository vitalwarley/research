from pathlib import Path

import cv2
import torch


class Sample:
    # TODO: move to utils.py
    NAME2LABEL = {
        # "non-kin": 0,
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

    def __init__(self, id: str, f1: str, f2: str, kin_relation: str, is_kin: str, *args, **kwargs):
        self.id = id
        self.f1 = f1
        self.f1fid = f1.split("/")[2]
        self.f2 = f2
        self.f2fid = f2.split("/")[2]
        self.kin_relation = kin_relation
        self.is_kin = int(is_kin)
        self.is_same_generation = self.kin_relation in ["bb", "ss", "sibs"]


class SampleKFC(Sample):
    NAME2LABEL = {
        "fs": 0,
        "fd": 1,
        "ms": 2,
        "md": 3,
        "fms": 4,
        "fmd": 5,
        "fsd": 6,
        "msd": 7,
    }

    def __init__(self, id: str, f1: str, f2: str, kin_relation: str, is_kin: str, race: str):
        super().__init__(id, f1, f2, kin_relation, is_kin)
        self.race = race


def one_hot_encode_kinship(relation):
    index = Sample.NAME2LABEL[relation] - 1
    one_hot = torch.zeros(len(Sample.NAME2LABEL))
    one_hot[index] = 1
    return one_hot


def read_image(path: Path, shape=(112, 112)):
    # TODO: add to utils.py
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    return img