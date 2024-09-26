from pathlib import Path

import lightning as L
from datasets.fiw import FIW, FIWFamily, FIWFamilyV2, FIWGallery, FIWPairs, FIWProbe, FIWSearchRetrieval
from datasets.utils import SampleGallery, SampleProbe, sr_collate_fn_v2
from torch.utils.data import DataLoader
from torchvision import transforms as T


class FIWFaCoRNet(FIW):

    TRAIN_PAIRS = "txt/train_sort_A2_m.txt"
    VAL_PAIRS_MODEL_SEL = "txt/val_choose_A.txt"
    VAL_PAIRS_THRES_SEL = "txt/val_A.txt"
    TEST_PAIRS = "txt/test_A.txt"

    # AdaFace uses BGR -- should I revert conversion read_image here?

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, item):
        images, labels = super().__getitem__(item)
        # Convert img1 and img2 to BGR - they're tensors (C, H, W)
        # img1 = img1[[2, 1, 0], :, :]
        # img2 = img2[[2, 1, 0], :, :]
        return images, labels


class FaCoRNetDataModule(L.LightningDataModule):
    DATASETS = {"facornet": FIWFaCoRNet, "facornet-family": FIWFamilyV2, "fiw-pairs": FIWPairs}

    def __init__(
        self, dataset: str, biased: bool, batch_size=20, root_dir=".", shuffle_validation=False, transform=None
    ):
        super().__init__()
        self.dataset = self.DATASETS[
            dataset
        ]  # Couldn't find a better way to do this; if I put into init, I can't make it work
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.transform = transform or T.Compose([T.ToTensor()])
        self.biased = biased
        self.shuffle_validation = shuffle_validation

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.TRAIN_PAIRS),
                batch_size=self.batch_size,
                biased=self.biased,
                transform=self.transform,
            )
            self.val_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.VAL_PAIRS_MODEL_SEL),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.VAL_PAIRS_THRES_SEL),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.TEST_PAIRS),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        print(f"Setup {stage} datasets")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.biased,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_validation,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_validation,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )


class FaCoRNetFamilyDataModule(FaCoRNetDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage=None):
        super().setup(stage)
        if stage == "fit" or stage is None:
            self.family_dataset = FIWFamily(
                root_dir=Path(self.root_dir, "images/Train_A/train-faces/"),
                uniform_family=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        family_dataloader = DataLoader(
            self.family_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        return [super().train_dataloader(), family_dataloader]


class FamilyDataModule(FaCoRNetDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage=None):
        super().setup(stage)
        if stage == "fit" or stage is None:
            self.family_dataset = FIWFamily(
                root_dir=Path(self.root_dir, "images/Train_A/train-faces/"),
                uniform_family=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(self.family_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)


class FaCoRNetDMTask3(L.LightningDataModule):

    def __init__(self, root_dir=".", batch_size=20, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform or T.Compose([T.ToTensor()])

    def setup(self, stage=None):
        if stage == "predict" or stage is None:
            self.probe_dataset = FIWProbe(
                root_dir=self.root_dir,
                sample_path="txt/probe.txt",
                sample_cls=SampleProbe,
                transform=self.transform,
            )
            self.gallery_dataset = FIWGallery(
                root_dir=self.root_dir,
                sample_path="txt/gallery.txt",
                sample_cls=SampleGallery,
                transform=self.transform,
            )
            self.search_retrieval = FIWSearchRetrieval(self.probe_dataset, self.gallery_dataset, self.batch_size)
        print(f"Setup {stage} datasets")

    def predict_dataloader(self):
        return DataLoader(
            self.search_retrieval,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=sr_collate_fn_v2,
            # Why num_workers reset the gallery_start_index?
            num_workers=4,
        )


class FaCoRNetDataModuleV2(L.LightningDataModule):
    """
    Contrastive Learning at kinship-level discrimination.
    """

    def __init__(self, biased: bool, batch_size=20, root_dir=".", transform=None, dataset: str = ""):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.transform = transform or T.Compose([T.ToTensor()])
        self.biased = biased

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = FIWPairs(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.TRAIN_PAIRS),
                batch_size=self.batch_size,
                biased=self.biased,
                transform=self.transform,
            )
            self.val_dataset = FIWFaCoRNet(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.VAL_PAIRS_MODEL_SEL),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = FIWFaCoRNet(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.VAL_PAIRS_THRES_SEL),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = FIWFaCoRNet(
                root_dir=self.root_dir,
                sample_path=Path(FIWFaCoRNet.TEST_PAIRS),
                batch_size=self.batch_size,
                transform=self.transform,
            )
        print(f"Setup {stage} datasets")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )


if __name__ == "__main__":
    # Test
    root_dir = "../datasets/rfiw2021-track3"
    data_module = FaCoRNetDMTask3(root_dir=root_dir)
    data_module.setup("predict")
    fiw_sr = data_module.predict_dataloader()
    print(len(fiw_sr))
    # Iters through the probe and gallery samples
    for i, (probe_index, probe_images, gallery_indexes, gallery_images) in enumerate(fiw_sr):
        # if i % len(fiw_gallery) == 0:
        print(probe_index, len(probe_images), gallery_indexes)
        if probe_index == 2:
            break
