import random
from itertools import chain, groupby
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import Sample, one_hot_encode_kinship, read_image


class MTCFDataset(Dataset):
    """
    Similar to Zhang et al. (2021)
    """

    def __init__(
        self,
        root_dir: Path,
        sample_path: Path,
        negatives_per_sample: int = 1,
        extend_with_same_gen: bool = True,
        transform=None,
    ):
        self.root_dir = root_dir
        self.sample_path = sample_path
        self.extend_with_same_gen = extend_with_same_gen
        self.transform = transform
        self.samples = self.load_sample()
        print(
            f"Loaded {len(self.samples)} samples from {sample_path} "
            + "(with duplicated samples for same generation bb, ss, sibs)."
        )
        if negatives_per_sample:
            self.add_negative_samples(negatives_per_sample)
            print(f"Added negative samples, now we have {len(self.samples)} samples.")

    def load_sample(self):
        sample_list = []
        lines = Path(self.sample_path).read_text().strip().split("\n")
        for line in lines:
            if len(line) < 1:
                continue
            tmp = line.split(" ")
            sample = Sample(tmp[0], tmp[1], tmp[2], tmp[-2], tmp[-1])
            sample_list.append(sample)
            if sample.is_same_generation and self.extend_with_same_gen:
                # Create new sample swapping f1 and f2
                sample_list.append(Sample(tmp[0], tmp[2], tmp[1], tmp[-2], tmp[-1]))
        return sample_list

    def add_negative_samples(self, negatives_per_sample: int):
        """
        Randomly swap x_2 with x_2 from `negatives_per_sample` different families with the same kinship label y.
        """
        # Group samples by kinship label, and within each label, group by family ID
        kinship_dict = {
            k: {fid: [s for s in samples if s.f1fid == fid] for fid in set(s.f1fid for s in samples)}
            for k, samples in [
                (k, list(samples))
                for k, samples in groupby(sorted(self.samples, key=lambda s: s.kin_relation), lambda s: s.kin_relation)
            ]
        }

        print(f"Adding {negatives_per_sample} negative samples per sample...")

        # Function to process each sample
        def process_sample(sample):
            kinship_group = kinship_dict[sample.kin_relation]
            other_families = {fid: members for fid, members in kinship_group.items() if fid != sample.f1fid}
            all_other_members = list(chain(*other_families.values()))

            if not all_other_members:
                return []

            new_samples = []
            for _ in range(negatives_per_sample):
                random_sample = random.choice(all_other_members)
                new_sample = Sample(sample.id, sample.f1, random_sample.f2, sample.kin_relation, False)
                new_samples.append(new_sample)

            return new_samples

        new_samples_lists = map(process_sample, self.samples)

        # Flatten list of lists and extend self.samples
        self.samples.extend(chain(*new_samples_lists))

    def _add_negative_samples(self, negatives_per_sample: int):
        """
        Randomly swap x_2 with x_2 from `negatives_per_sample` different families with the same kinship label y.
        """
        new_samples = []
        # Create a dict with keys being kinship labels and values being a list of samples with that kinship label
        kinship_dict = {k: [s for s in self.samples if s.kin_relation == k] for k in Sample.NAME2LABEL.keys()}
        print(f"Adding {negatives_per_sample} negative samples per sample...")
        for sample in tqdm(self.samples):
            # Randomly select `negatives_per_sample` samples
            added_samples = 0
            while added_samples < negatives_per_sample:
                # Randomly select a sample with the same kinship label
                random_sample = np.random.choice(kinship_dict[sample.kin_relation])
                # Check if the random sample comes from a different family
                if sample.f1fid != random_sample.f2fid:
                    # Create new sample
                    new_sample = Sample(sample.id, sample.f1, random_sample.f2, sample.kin_relation, sample.is_kin)
                    # Swap f2
                    new_samples.append(new_sample)
                    added_samples += 1
        self.samples.extend(new_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        # id, f1, f2, kin_relation, is_kin
        sample = self.samples[item]
        img1, img2 = read_image(self.root_dir / sample.f1), read_image(self.root_dir / sample.f2)
        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)
        is_kin = torch.tensor(int(sample.is_kin))
        kin_1hot = one_hot_encode_kinship(sample.kin_relation)
        labels = (kin_1hot, is_kin)
        return img1, img2, labels


# Test dataset as script
if __name__ == "__main__":
    from torchvision import transforms

    transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            # left-right filliping, random contrast, brightness, saturation with a probability of 0.5
            transforms.RandomApply(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
        ]
    )

    HERE = Path(__file__).parent.parent.parent

    def get_set_info(set: str):
        rfiw2021_root = HERE / "rfiw2021/Track1"
        sample_path = rfiw2021_root / f"sample0/{set}.txt"
        dataset = MTCFDataset(
            root_dir=rfiw2021_root,
            sample_path=sample_path,
            transform=transforms,
        )
        print(len(dataset))
        print(dataset[0])
        # Count frequency of each kinship relation
        from collections import Counter

        counter = Counter()
        for sample in dataset.samples:
            counter[sample.kin_relation] += 1
        for k, v in counter.items():
            print(k, v)

    get_set_info("train_sort")
    get_set_info("val_choose")
    get_set_info("val")
