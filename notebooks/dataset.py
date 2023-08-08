import glob
import os
import random

import numpy as np
from keras.preprocessing.image import load_img
from torch.utils.data import Dataset
from torchvision import transforms

from rfiw2021.Track1.utils import np2tensor


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
        for family_id in self.families:
            family_path = os.path.join(self.root_dir, f"F0{family_id}")  # FIXME: F0{family_id} is a hack
            # shuffle, then select member_limit members
            member_ids = os.listdir(family_path)
            print(f"Family {family_id} has {len(member_ids)} members")
            # np.random.shuffle(member_ids)
            # member_ids = member_ids[: self.member_limit]
            # get one image per member
            for member_id in member_ids:
                member_path = os.path.join(family_path, member_id)
                member_images = glob.glob(f"{member_path}/*.jpg")
                print(f"Member {member_id} has {len(member_images)} images")
                # randomly select one image per member
                if len(member_images) > self.samples_per_member:
                    member_images = random.sample(member_images, self.samples_per_member)
                # select all images per member
                for image in member_images:
                    sample_list.append((image, family_id))

        print(f"Total samples: {len(sample_list)}")
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        img = load_img(path, target_size=(112, 112))
        return img

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        sample, family_id = self.sample_list[item]
        img = self.read_image(sample)
        if self.transform is not None:
            img = self.transform(img)
        img = np2tensor(self.preprocess(np.array(img, dtype=float)))
        return img, family_id
