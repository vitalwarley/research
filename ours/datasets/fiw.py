from collections import defaultdict
from itertools import combinations, islice
from pathlib import Path

import cv2
import torch
from datasets.utils import Sample, SampleGallery, SampleProbe, sr_collate_fn_v2
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms as T
from tqdm import tqdm


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
        (img1, img2) = self._process_images(sample)
        labels = self._process_labels(sample)
        sample = (img1, img2, labels)
        return sample


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


class FIWFamilyV2(FIW):
    """
    Originally FIWFaCoRNetFamily.
    """

    # FaCoRNet dataset
    TRAIN_PAIRS = "txt/train_sort_A2_m.txt"
    VAL_PAIRS_MODEL_SEL = "txt/val_choose_A.txt"
    VAL_PAIRS_THRES_SEL = "txt/val_A.txt"
    TEST_PAIRS = "txt/test_A.txt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Enconde all samples f1fid and f2fid to set of unique values
        self.fid_set = set()
        for sample in self.sample_list:
            self.fid_set.add(sample.f1fid)
            self.fid_set.add(sample.f2fid)
        # Map each fid to an index
        self.fid_set = sorted(list(self.fid_set))
        self.fid2idx = {fid: idx for idx, fid in enumerate(self.fid_set)}

    def _process_labels(self, sample):
        is_kin = torch.tensor(sample.is_kin)
        kin_id = self.sample_cls.NAME2LABEL[sample.kin_relation]
        fid1, fid2 = int(sample.f1fid), int(sample.f2fid)
        # Get index for each fid
        fid1, fid2 = torch.tensor(self.fid2idx[fid1]), torch.tensor(self.fid2idx[fid2])
        labels = (kin_id, is_kin, (fid1, fid2))
        return labels


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
        self.persons = []
        self.persons2idx = {}
        self.idx2persons = {}
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
            persons.append(sample.f1)
            persons.append(sample.f2)
            if relation not in unique_relations:
                unique_relations.add(relation)
                person1_images = list(Path(self.root_dir, self.images_dir, sample.f1).parent.glob("*.jpg"))
                person2_images = list(Path(self.root_dir, self.images_dir, sample.f2).parent.glob("*.jpg"))
                # Filter path relative to images_dir
                person1_images = [str(img.relative_to(self.root_dir / self.images_dir)) for img in person1_images]
                person2_images = [str(img.relative_to(self.root_dir / self.images_dir)) for img in person2_images]
                # Filter relative to self.filepaths; apparently some images are missing in the train.csv
                person1_images = [person for person in person1_images if person in self.filepaths]
                person2_images = [person for person in person2_images if person in self.filepaths]
                if person1_images and person2_images:
                    labels = (sample.f1mid, sample.f2mid, sample.f1fid)
                    relationships.append((person1_images, person2_images, labels))
                    self.fam2rel[sample.f1fid].append(len(relationships) - 1)

        self.person2family = {person: int(person.split("/")[2][1:]) for person in persons}
        self.persons = sorted(list(set(persons)))
        self.persons2idx = {person: idx for idx, person in enumerate(self.persons)}
        self.idx2persons = {idx: person for person, idx in self.persons2idx.items()}

        print(f"Generated {len(relationships)} relationships")
        print(f"Found {len(self.persons)} unique persons")

        return relationships

    def _process_one_image(self, image_path):
        if image_path in self.cache:
            return self.cache[image_path]
        image = super()._process_one_image(image_path)
        self.cache[image_path] = image
        return image

    def __getitem__(self, idx: list[tuple[int, int]]):
        img1_idx, img2_idx = list(zip(*idx))
        imgs1, imgs2 = [self.persons[idx] for idx in img1_idx], [self.persons[idx] for idx in img2_idx]
        is_kin = [
            int(self.person2family[person1] == self.person2family[person2]) for person1, person2 in zip(imgs1, imgs2)
        ]
        imgs1 = [self._process_one_image(img) for img in imgs1]
        imgs2 = [self._process_one_image(img) for img in imgs2]
        sample = (imgs1, imgs2, is_kin)  # collate!
        return sample

    def __len__(self):
        return len(self.relationships)


class FIWPairs(FIW):

    # FaCoRNet dataset
    TRAIN_PAIRS = "txt/train_sort_A2_m.txt"
    VAL_PAIRS_MODEL_SEL = "txt/val_choose_A.txt"
    VAL_PAIRS_THRES_SEL = "txt/val_A.txt"
    TEST_PAIRS = "txt/test_A.txt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_combs = 10000
        self.sample_list = self.create_kinship_pairs_list()

    def create_kinship_pairs_list(self):
        print(f"Creating kinship pairs list from {self.sample_path}")
        # Group samples by kinship type
        grouped_by_kinship = {}
        for sample in tqdm(self.sample_list):
            kinship_type = sample.kin_relation
            if kinship_type not in grouped_by_kinship:
                grouped_by_kinship[kinship_type] = []
            grouped_by_kinship[kinship_type].append(sample)

        # For each kinship type, create pairs of samples
        progress_bar = tqdm(total=len(grouped_by_kinship) * self.num_combs)
        all_samples_by_type = []
        for kinship_type, samples in grouped_by_kinship.items():
            kinship_samples = []
            # Using combinations to create unique pairs within the same kinship type
            for pair1, pair2 in islice(combinations(samples, 2), self.num_combs):
                # Creating the desired format ((pair1_img1, pair1_img2), (pair2_img1, pair2_img2), kinship_type)
                new_sample = (pair1, pair2, kinship_type)
                kinship_samples.append(new_sample)
                progress_bar.update(1)
            all_samples_by_type.append(kinship_samples)

        progress_bar.close()

        new_samples_list = []
        progress_bar = tqdm(total=len(all_samples_by_type) * self.num_combs)

        while True:
            added_any = False
            for kinship_samples in all_samples_by_type:
                if kinship_samples:
                    new_samples_list.append(kinship_samples.pop(0))
                    added_any = True
                    progress_bar.update(1)
            if not added_any:
                break

        return new_samples_list

    def __getitem__(self, item):
        # id, f1, f2, kin_relation, is_kin
        sample = self.sample_list[item]
        pair1, pair2, kinship_type = sample
        (p1_im1, p1_im2) = self._process_images(pair1)
        (p2_im1, p2_im2) = self._process_images(pair2)
        p1_is_kin = self._process_labels(pair1)[1]
        p2_is_kin = self._process_labels(pair2)[1]
        label = self.sample_cls.NAME2LABEL[kinship_type]
        sample = ((p1_im1, p1_im2), (p2_im1, p2_im2), (p1_is_kin, p2_is_kin, label))
        return sample


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
        # Calculate probe index and gallery index
        probe_index = idx // len(self.gallery_samples)
        gallery_index = idx % len(self.gallery_samples)

        # Get the probe and gallery information
        probe_id, probe_images = self.probe_samples[probe_index]
        gallery_id, gallery_image = self.gallery_samples[gallery_index]

        # Return a tuple of probe and gallery data
        return (probe_id, probe_images), (gallery_id, gallery_image)


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
    fiw_sr = FIWSearchRetrieval(fiw_probe, fiw_gallery)
    print(len(fiw_sr))
    # Create a gallery dataloader and test them
    sr_loader = torch.utils.data.DataLoader(fiw_sr, batch_size=1, shuffle=False, collate_fn=sr_collate_fn_v2)
    # Iters through the probe and gallery samples
    for i, ((probe_index, probe_images), (gallery_indexes, gallery_images)) in enumerate(sr_loader):
        # if i % len(fiw_gallery) == 0:
        print(probe_index, len(probe_images), gallery_indexes)
        if i > 2:
            break
