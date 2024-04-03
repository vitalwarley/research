from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .fiw import FIW


class FIWFaCoRNet(FIW):

    TRAIN_PAIRS = "txt/train_sort_A2_m.txt"
    VAL_PAIRS_MODEL_SEL = "txt/val_choose_A.txt"
    VAL_PAIRS_THRES_SEL = "txt/val_A.txt"
    TEST_PAIRS = "txt/test_A.txt"

    # AdaFace uses BGR -- should I revert conversion read_image here?

    def __init__(self, batch_size: int, **kwargs):
        self.batch_size = batch_size
        super().__init__(**kwargs)

    def __getitem__(self, item):
        (img1, img2, labels) = super().__getitem__(item)
        # Convert img1 and img2 to BGR - they're tensors (C, H, W)
        # img1 = img1[[2, 1, 0], :, :]
        # img2 = img2[[2, 1, 0], :, :]
        return img1, img2, labels


class FaCoRNetDataModule(L.LightningDataModule):
    def __init__(self, batch_size=20, root_dir=".", transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.transform = transform or T.Compose([T.ToTensor()])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = FIW(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.TRAIN_PAIRS),
                batch_size=self.batch_size,
                biased=True,
                transform=self.transform,
            )
            self.val_dataset = FIW(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.VAL_PAIRS_MODEL_SEL),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = FIW(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.VAL_PAIRS_THRES_SEL),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = FIW(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.TEST_PAIRS),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        print(f"Setup {stage} datasets")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
