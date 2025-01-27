import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def np2tensor(arrays, device="gpu", dtype=torch.float):
    tensor = torch.from_numpy(arrays).type(dtype)
    return tensor.cuda() if device == "gpu" else tensor


class FIW(Dataset):
    def __init__(
        self,
        root_dir: str = "",
        transform: transforms.Compose | None = None,
        families: list = [],
        member_limit: int = 10,
        samples_per_member: int = 1,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.families = [str(i) for i in families]
        self.member_limit = member_limit
        self.samples_per_member = samples_per_member
        self.sample_list = self.load_sample()

    def load_sample(self):
        sample_list = []

        if not self.families:
            self.families = os.listdir(self.root_dir)

        for family_id in tqdm(self.families):
            if "F" not in family_id:  # hack
                family_path = os.path.join(
                    self.root_dir, f"F{int(family_id):04}"
                )  # FIXME: F0{family_id} is a hack
            else:
                family_path = os.path.join(
                    self.root_dir, family_id
                )  # FIXME: F0{family_id} is a hack
            # shuffle, then select member_limit members
            member_ids = os.listdir(family_path)
            # np.random.shuffle(member_ids)
            # member_ids = member_ids[: self.member_limit]
            # get one image per member
            for member_id in member_ids:
                member_path = os.path.join(family_path, member_id)
                member_images = glob.glob(f"{member_path}/*.jpg")
                # print(f"Member {member_id} has {len(member_images)} images")
                # randomly select one image per member
                if len(member_images) > self.samples_per_member:
                    member_images = random.sample(
                        member_images, self.samples_per_member
                    )
                # select all images per member
                for image in member_images:
                    sample_list.append((image, family_id))

        print(f"Total samples: {len(sample_list)}")
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        # TODO: add to utils.py
        # image_path = f"{self.root_dir / self.images_dir}/{path}"
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        return img

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        sample, family_id = self.sample_list[item]
        img = self.read_image(sample)
        if self.transform is not None:
            img = self.transform(img)
        img = np2tensor(self.preprocess(np.array(img, dtype=float)))
        return item, img, family_id


class FIWPair(Dataset):
    def __init__(
        self,
        root_dir: str = "",
        csv_path: str = "",
        transform: transforms.Compose | None = None,
        families: list = [],
        member_limit: int = 10,
        samples_per_member: int = 1,
    ):
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.transform = transform
        self.families = families
        self.member_limit = member_limit
        self.samples_per_member = samples_per_member
        self.sample_list = self.load_sample()
        print(f"Total samples: {len(self.sample_list)}")

    def load_sample(self):
        # Read from csv_path, delimted by " "; use the header to 'id face1_path face2_path kin_relation is_kin'
        data = pd.read_csv(
            self.csv_path,
            delimiter=" ",
            header=None,
            names=["id", "face1_path", "face2_path", "kin_relation", "is_kin"],
        )
        # Check data is loaded correctly
        assert len(data) > 0, "No data loaded from csv_path"
        # Consider that face1_path and face2_path has the following format:
        # <root-dir>/F{family_id}/{member_id}/{image_name}.jpg
        # We need to extract the family_id from the path
        data.loc[:, "face1_family_id"] = (
            data["face1_path"]
            .apply(lambda x: x.split("/")[-3].replace("F", ""))
            .astype(int)
        )
        data.loc[:, "face2_family_id"] = (
            data["face2_path"]
            .apply(lambda x: x.split("/")[-3].replace("F", ""))
            .astype(int)
        )
        # Filter the data by family_id
        if self.families:
            data = data[
                data["face1_family_id"].isin(self.families)
                | data["face2_family_id"].isin(self.families)
            ]
            assert (
                len(data) > 0
            ), "No data loaded from csv_path after filtering by families list: {}".format(
                self.families
            )
        # Drop id column, reset index
        data = data.drop(columns=["id"]).reset_index(drop=True)
        # TODO: make programaticly
        # Filter rows where face1_family_id == face2_family_id == 250
        # data = data[((data["face1_family_id"] == 250) & (data["face2_family_id"] == 250))]
        # Count samples by kin_relation
        # print(data.groupby("kin_relation").count())
        return data

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        # TODO: add to utils.py
        # image_path = f"{self.root_dir / self.images_dir}/{path}"
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        return img

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        sample = self.sample_list.iloc[item]
        face1 = self.read_image(str(self.root_dir / sample.face1_path))
        face2 = self.read_image(str(self.root_dir / sample.face2_path))
        if self.transform is not None:
            face1 = self.transform(face1)
            face2 = self.transform(face2)

        face1 = np2tensor(self.preprocess(np.array(face1, dtype=float)))
        face2 = np2tensor(self.preprocess(np.array(face2, dtype=float)))

        return (
            face1,
            face2,
            sample.kin_relation,
            sample.face1_family_id,
            sample.face2_family_id,
            sample.is_kin,
        )
