import random
from pathlib import Path
from typing import Optional

import cv2
import lightning as L
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class KinFaceWDataset(Dataset):
    def __init__(self, root_dir: Path, dataset: str = "I", transform=None):
        self.root_dir = root_dir
        self.dataset = dataset
        self.kinship_types = {"father-dau": "fd", "father-son": "fs", "mother-dau": "md", "mother-son": "ms"}
        self.transform = transform
        self.data = self._load_data()
        self.pairs = self._construct_pairs()

    def _load_data(self):
        dataset_dir = self.root_dir / f"KinFaceW-{self.dataset}" / "images"
        print(f"Loading data from {dataset_dir}...")
        data = {kt: [] for kt in self.kinship_types}

        for folder, kt in self.kinship_types.items():
            kinship_dir = dataset_dir / folder
            images = list(kinship_dir.glob("*.jpg"))
            data[kt] = images

        return data

    def _construct_pairs(self):
        positive_pairs = []
        negative_pairs = []

        for kinship in self.kinship_types.values():
            pairs = {}
            for img in self.data[kinship]:
                pair_id, member_id = img.stem.split("_")[1:]
                if pair_id not in pairs:
                    pairs[pair_id] = {}
                pairs[pair_id][member_id] = img

            # Construct positive pairs
            for pair_id, members in pairs.items():
                if len(members) == 2:
                    positive_pairs.append((members["1"], members["2"], kinship, 1))

            # Construct negative pairs
            parents = [members["1"] for members in pairs.values() if "1" in members]
            children = [members["2"] for members in pairs.values() if "2" in members]

            for parent in parents:
                true_child = next(
                    members["2"] for members in pairs.values() if "1" in members and members["1"] == parent
                )
                false_children = [child for child in children if child != true_child]
                if false_children:
                    false_child = random.choice(false_children)
                    negative_pairs.append((parent, false_child, kinship, 0))

        print(f"# positive pairs = {len(positive_pairs)}, # negative pairs = {len(negative_pairs)}")

        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        parent_img_path, child_img_path, kinship, is_kin = self.pairs[idx]

        parent_img = cv2.imread(str(parent_img_path))
        parent_img = cv2.cvtColor(parent_img, cv2.COLOR_BGR2RGB)
        if self.transform:
            parent_img = self.transform(parent_img)

        child_img = cv2.imread(str(child_img_path))
        child_img = cv2.cvtColor(child_img, cv2.COLOR_BGR2RGB)
        if self.transform:
            child_img = self.transform(child_img)

        kinship_label = ["fd", "fs", "md", "ms"].index(kinship)

        return parent_img, child_img, (kinship_label, is_kin)  # to be compatible with FIW


class KinFaceWDatasetKFold(Dataset):
    def __init__(self, root_dir: Path, dataset: str = "I", fold: int = 1, is_train: bool = True, transform=None):
        self.root_dir = root_dir
        self.dataset = dataset
        self.fold = fold
        self.is_train = is_train
        self.kinship_types = {"father-dau": "fd", "father-son": "fs", "mother-dau": "md", "mother-son": "ms"}
        self.transform = transform
        self.data = self._load_data()
        self.pairs = self._construct_pairs()

    def _load_data(self):
        dataset_dir = self.root_dir / f"KinFaceW-{self.dataset}" / "images"
        data = {kt: [] for kt in self.kinship_types.values()}

        fold_ranges = {
            "I": {
                "fd": [[1, 27], [28, 54], [55, 81], [82, 108], [109, 134]],
                "fs": [[1, 31], [32, 64], [65, 96], [97, 124], [125, 156]],
                "md": [[1, 25], [26, 50], [51, 75], [76, 101], [102, 127]],
                "ms": [[1, 23], [24, 46], [47, 69], [70, 102], [93, 116]],
            },
            "II": {"all": [[1, 50], [51, 100], [101, 150], [151, 200], [201, 250]]},
        }

        for folder, kinship in self.kinship_types.items():
            kinship_dir = dataset_dir / folder
            images = list(kinship_dir.glob("*.jpg"))
            print(f"{kinship} = {len(images)}")

            for img in images:
                pair_id = int(img.stem.split("_")[1])
                fold_index = next(
                    i
                    for i, range in enumerate(
                        fold_ranges["I"][kinship] if self.dataset == "I" else fold_ranges["II"]["all"]
                    )
                    if range[0] <= pair_id <= range[1]
                )

                if self.is_train and fold_index != self.fold - 1:
                    data[kinship].append(img)
                elif not self.is_train and fold_index == self.fold - 1:
                    data[kinship].append(img)

        return data

    def _construct_pairs(self):
        positive_pairs = []
        negative_pairs = []

        for kin_label in self.kinship_types.values():
            pairs = {}
            for img in self.data[kin_label]:
                pair_id, member_id = img.stem.split("_")[1:]
                if pair_id not in pairs:
                    pairs[pair_id] = {}
                pairs[pair_id][member_id] = img

            # Construct positive pairs
            for pair_id, members in pairs.items():
                if len(members) == 2:
                    positive_pairs.append((members["1"], members["2"], kin_label, 1))

            # Construct negative pairs
            parents = [members["1"] for members in pairs.values() if "1" in members]
            children = [members["2"] for members in pairs.values() if "2" in members]

            if not self.is_train:
                for parent in parents:
                    true_child = next(
                        members["2"] for members in pairs.values() if "1" in members and members["1"] == parent
                    )
                    false_children = [child for child in children if child != true_child]
                    if false_children:
                        false_child = random.choice(false_children)
                        negative_pairs.append((parent, false_child, kin_label, 0))

        print(f"# positive pairs = {len(positive_pairs)}, # negative pairs = {len(negative_pairs)}")

        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        parent_img_path, child_img_path, kinship, is_kin = self.pairs[idx]

        parent_img = self._load_image(parent_img_path)
        if self.transform:
            parent_img = self.transform(parent_img)
        child_img = self._load_image(child_img_path)
        if self.transform:
            child_img = self.transform(child_img)

        kinship_label = list(self.kinship_types.values()).index(kinship)
        return (parent_img, child_img), (kinship_label, is_kin)

    def _load_image(self, img_path):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class KinFaceWDataModule(L.LightningDataModule):
    def __init__(self, root_dir: Path, dataset: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.root_dir = root_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((112, 112)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def setup(self, stage: Optional[str] = None):
        self.val_dataset = KinFaceWDataset(self.root_dir, self.dataset, transform=self.transform)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class KinFaceWDataModuleKFold(L.LightningDataModule):
    def __init__(self, root_dir: Path, dataset: str, fold: int, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.root_dir = root_dir
        self.fold = fold
        self.dataset = dataset
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=100)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = KinFaceWDatasetKFold(
            self.root_dir, self.dataset, self.fold, is_train=True, transform=self.transform
        )
        self.val_dataset = KinFaceWDatasetKFold(
            self.root_dir, self.dataset, self.fold, is_train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# Example usage
if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent / "datasets/kinfacew"
    dataset = "I"  # or 'II'

    data_module = KinFaceWDataModule(root_dir, dataset, 1)
    data_module.setup()

    # Access the dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
