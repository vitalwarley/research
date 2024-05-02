from pathlib import Path

import cv2
import numpy as np
import torch
from datasets.utils import Sample, SampleGallery, SampleProbe
from torch.utils.data import Dataset, IterableDataset


class FIW(Dataset):
    def __init__(
        self,
        root_dir: str = "",
        sample_path: str | Path = "",
        batch_size: int = 1,
        biased: bool = False,
        transform=None,
        sample_cls=Sample,
    ):
        self.root_dir = Path(root_dir)
        self.images_dir = "images"
        self.sample_path = Path(sample_path)
        self.batch_size = batch_size
        self.transform = transform
        self.bias = 0
        self.biased = biased
        self.sample_cls = sample_cls
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


class FIWSearchRetrieval(Dataset):
    def __init__(self, probe_dataset, gallery_dataset):
        super().__init__()
        self.probe_dataset = probe_dataset
        self.probe_samples = list(iter(probe_dataset))  # Cache probe samples

        self.gallery_dataset = gallery_dataset
        self.gallery_samples = list(iter(gallery_dataset))  # Cache gallery samples
        self.gallery_start_index = 0  # Initialize gallery start index

    def __len__(self):
        # Return the total number of possible combinations of probe and gallery images
        return len(self.probe_samples) * len(self.gallery_samples)

    def __getitem__(self, idx):
        # Calculate probe index
        probe_index = idx % len(self.probe_samples)
        probe_id, probe_images = self.probe_samples[probe_index]
        num_probe_images = len(probe_images)

        # Calculate dynamic gallery indices
        gallery_ids = []
        gallery_images = []
        for i in range(num_probe_images):
            current_gallery_index = (self.gallery_start_index + i) % len(self.gallery_samples)
            gallery_id, gallery_image = self.gallery_samples[current_gallery_index]
            gallery_ids.append(gallery_id)
            gallery_images.append(gallery_image)

        # Update gallery_start_index for the next probe
        self.gallery_start_index = (self.gallery_start_index + num_probe_images) % len(self.gallery_samples)

        # TODO: adjust for when len(gallery_samples) < len(probe_samples); maybe repeat in-batch samples?

        return ((probe_id, probe_images), (gallery_ids, gallery_images))


class FIWProbe(FIW, IterableDataset):

    def __iter__(self):
        self.sample_iter = iter(self.sample_list)  # Reset iterator
        return self

    def __next__(self):
        sample = next(self.sample_iter)  # Get next sample
        imgs = self._read_dir(sample.s1_dir)
        return sample.id, imgs

    def _read_dir(self, dir):
        images = []
        for image in Path(self.root_dir, self.images_dir, dir).iterdir():
            image = image.relative_to(self.root_dir / self.images_dir)
            image = self.read_image(image)  # Assuming this is defined
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return images


class FIWGallery(FIW, IterableDataset):

    def __iter__(self):
        self.sample_iter = iter(self.sample_list)  # Reset iterator
        return self

    def __next__(self):
        sample = next(self.sample_iter)
        img = self.read_image(sample.f1)  # This needs to be defined
        if self.transform:
            img = self.transform(img)
        return sample.id, img


if __name__ == "__main__":
    # Test FIWProbe and FIWGallery
    root_dir = "../datasets/rfiw2021-track3"
    probe_path = "txt/probe.txt"
    gallery_path = "txt/gallery.txt"
    fiw_probe = FIWProbe(root_dir=root_dir, sample_path=probe_path, sample_cls=SampleProbe)
    fiw_gallery = FIWGallery(root_dir=root_dir, sample_path=gallery_path, sample_cls=SampleGallery)
    fiw_sr = FIWSearchRetrieval(fiw_probe, fiw_gallery)
    print(len(fiw_sr))
    # Create a gallery dataloader and test them
    sr_loader = torch.utils.data.DataLoader(fiw_sr, batch_size=5, shuffle=False)
    # Iters through the probe and gallery samples
    for i, ((probe_index, probe_images), (gallery_indexes, gallery_images)) in enumerate(sr_loader):
        # if i % len(fiw_gallery) == 0:
        print(probe_index, len(probe_images), gallery_indexes)
        if i > 2:
            break
