import math
from collections import defaultdict
from itertools import combinations, islice
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms as T
from tqdm import tqdm

from datasets.utils import Sample, SampleGallery, SampleProbe, SampleTask2, sr_collate_fn_v2


class FIW(Dataset):

    TRAIN_PAIRS = "txt/train_sort_A2_m.txt"
    VAL_PAIRS_MODEL_SEL = "txt/val_choose_A.txt"
    VAL_PAIRS_THRES_SEL = "txt/val_A.txt"
    TEST_PAIRS = "txt/test_A.txt"

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
        self.transform = transform or T.Compose([T.ToTensor()])
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
        image_path = f"{self.root_dir / self.images_dir}/{path}"
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        return img

    def set_bias(self, bias):
        if self.biased:
            self.bias = bias

    def _process_images(self, sample):
        img1 = self._process_one_image(sample.f1)
        img2 = self._process_one_image(sample.f2)
        return img1, img2

    def _process_one_image(self, image_path):
        image = self.read_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

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
        images = self._process_images(sample)
        labels = self._process_labels(sample)
        sample = (images, labels)
        return sample


class FIWTask2(FIW):

    TRAIN_PAIRS = "txt/train.txt"
    VAL_PAIRS = "txt/val.txt"
    TEST_PAIRS = "txt/test.txt"
    SAMPLE = SampleTask2

    def _process_images(self, sample):
        img1 = self._process_one_image(sample.f1)
        img2 = self._process_one_image(sample.f2)
        img3 = self._process_one_image(sample.f3)
        return img1, img2, img3


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


class FIWFamilyV3(FIW):
    """
    To be used with the KinshipBatchSampler.
    """

    # FaCoRNet dataset
    TRAIN_PAIRS = "txt/train_sort_A2_m.txt"
    VAL_PAIRS_MODEL_SEL = "txt/val_choose_A.txt"
    VAL_PAIRS_THRES_SEL = "txt/val_A.txt"
    TEST_PAIRS = "txt/test_A.txt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Enconde all samples f1fid and f2fid to set of unique values
        self.fids = []
        self.filepaths = []
        for sample in self.sample_list:
            self.fids.append(sample.f1fid)
            self.fids.append(sample.f2fid)
            self.filepaths.append(sample.f1)
            self.filepaths.append(sample.f2)
        self.filepaths = set(self.filepaths)
        # Map each fid to an index
        self.fids = sorted(list(set(self.fids)))
        print(f"Found {len(self.fids)} unique fids")
        self.fid2idx = {fid: idx for idx, fid in enumerate(self.fids)}

        whitelist_dir = "MID"
        self.families = [
            [
                cur_person
                for cur_person in cur_family.iterdir()
                if cur_person.is_dir() and cur_person.name.startswith(whitelist_dir)
            ]
            for cur_family in self.root_dir.iterdir()
            if cur_family.is_dir()
        ]
        self.fam2rel = defaultdict(list)
        self.people = []
        self.person2idx = {}
        self.idx2person = {}
        self.person2family = {}
        self.cache = {}
        self.relationships = self._generate_relationships()

    def _generate_relationships(self):
        relationships = []
        unique_relations = set()
        persons = []
        for sample_idx, sample in enumerate(self.sample_list):
            # Only consider training set, therefore only positive samples
            relation = (sample.f1fid,) + tuple(sorted([sample.f1mid, sample.f2mid]))
            # Path to images f1 and f2
            persons.append(sample.f1)
            persons.append(sample.f2)
            if relation not in unique_relations:
                unique_relations.add(relation)  # New relation
                # Get all images from individuals in this relation (FX/MY/<images>)
                person1_images = list(Path(self.root_dir, self.images_dir, sample.f1).parent.glob("*.jpg"))
                person2_images = list(Path(self.root_dir, self.images_dir, sample.f2).parent.glob("*.jpg"))
                # Filter path relative to images_dir
                person1_images = [str(img.relative_to(self.root_dir / self.images_dir)) for img in person1_images]
                person2_images = [str(img.relative_to(self.root_dir / self.images_dir)) for img in person2_images]
                # Filter relative to self.filepaths; apparently some images are missing in the train.csv
                person1_images = [person for person in person1_images if person in self.filepaths]
                person2_images = [person for person in person2_images if person in self.filepaths]
                if person1_images and person2_images:
                    # Store pair and individual information: members IDs, family id, and kinship relation
                    labels = (sample.f1mid, sample.f2mid, sample.f1fid, sample.is_kin, sample.kin_relation)
                    # List with all images from both individuals
                    relationships.append((person1_images, person2_images, labels))
                    # Store index of the current relationship in the relationships list for a given family
                    self.fam2rel[sample.f1fid].append(len(relationships) - 1)
        # List of unique persons (based on the filepath)
        self.people = sorted(list(set(persons)))
        # Mapping from person to its index
        self.person2idx = {person: idx for idx, person in enumerate(self.people)}

        print(f"Generated {len(relationships)} relationships")
        print(f"Found {len(self.people)} unique persons")

        return relationships

    def _process_one_image(self, image_path):
        if image_path in self.cache:
            return self.cache[image_path]
        image = super()._process_one_image(image_path)
        self.cache[image_path] = image
        return image

    def __getitem__(self, idx: list[tuple[int, int]]):
        # idx comes from person indices (person2idx)
        img1_idx, img2_idx, labels = list(zip(*idx))
        imgs1, imgs2 = [self.people[idx] for idx in img1_idx], [self.people[idx] for idx in img2_idx]
        # Get is_kin from the stored relationship
        is_kin = [labels[i][3] == labels[i][3] for i in range(len(imgs1))]
        # Get kin_id from the stored relationship
        kin_ids = [labels[i][4] for i in range(len(imgs1))]  # collate fn will convert from name to label

        imgs1 = [self._process_one_image(img) for img in imgs1]
        imgs2 = [self._process_one_image(img) for img in imgs2]
        images = (imgs1, imgs2)
        labels = (kin_ids, is_kin)
        sample = (images, labels)
        return sample

    def __len__(self):
        return len(self.relationships)


class FIWSearchRetrieval(Dataset):
    def __init__(self, probe_dataset, gallery_dataset, batch_size=100):
        super().__init__()
        self.probe_dataset = probe_dataset
        self.probe_samples = list(iter(probe_dataset))  # Cache probe samples
        self.gallery_dataset = gallery_dataset
        self.gallery_samples = list(iter(gallery_dataset))  # Cache gallery samples
        self.batch_size = batch_size
        self.data = self._create_data_mapping()

    def _create_data_mapping(self):
        data = []
        num_batches = math.ceil(len(self.gallery_samples) / self.batch_size)
        print(f"# batches for each probe = {num_batches}")

        for probe_id, probe_images in self.probe_samples:
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(self.gallery_samples))
                gallery_batch = self.gallery_samples[start_idx:end_idx]
                data.append((probe_id, probe_images, gallery_batch))

        print(f"Total number of batches = {len(data)}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        probe_id, probe_images, gallery_batch = self.data[idx]
        return (probe_id, probe_images), gallery_batch


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
    fiw_probe = FIWProbe(
        root_dir=root_dir, sample_path=probe_path, sample_cls=SampleProbe, transform=T.Compose([T.ToTensor()])
    )
    fiw_gallery = FIWGallery(
        root_dir=root_dir, sample_path=gallery_path, sample_cls=SampleGallery, transform=T.Compose([T.ToTensor()])
    )
    fiw_sr = FIWSearchRetrieval(fiw_probe, fiw_gallery, 20)
    print(len(fiw_sr))
    # Create a gallery dataloader and test them
    sr_loader = DataLoader(fiw_sr, batch_size=1, shuffle=False, collate_fn=sr_collate_fn_v2)
    # Iters through the probe and gallery samples
    for i, (probe_index, probe_images, gallery_indexes, gallery_images) in enumerate(sr_loader):
        # if i % len(fiw_gallery) == 0:
        print(probe_index, len(probe_images), gallery_indexes)
        if i > 2:
            break
