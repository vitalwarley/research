import lightning as L
import numpy as np
from datasets.fiw import FIW
from datasets.utils import SampleKFC
from torch.utils.data import DataLoader
from torchvision import transforms as T

RACE_DICT = {
    "AA": np.array([0], dtype=np.int_),
    "A": np.array([1], dtype=np.int_),
    "C": np.array([2], dtype=np.int_),
    "I": np.array([3], dtype=np.int_),
    "AA&AA": np.array([4], dtype=np.int_),
    "AA&A": np.array([5], dtype=np.int_),
    "AA&C": np.array([6], dtype=np.int_),
    "AA&I": np.array([7], dtype=np.int_),
    "A&AA": np.array([8], dtype=np.int_),
    "A&A": np.array([9], dtype=np.int_),
    "A&C": np.array([10], dtype=np.int_),
    "A&I": np.array([11], dtype=np.int_),
    "C&AA": np.array([12], dtype=np.int_),
    "C&A": np.array([13], dtype=np.int_),
    "C&C": np.array([14], dtype=np.int_),
    "C&I": np.array([15], dtype=np.int_),
    "I&AA": np.array([16], dtype=np.int_),
    "I&A": np.array([17], dtype=np.int_),
    "I&C": np.array([18], dtype=np.int_),
    "I&I": np.array([19], dtype=np.int_),
}


class FIWKFC(FIW):

    TRAIN_PAIRS = "txt/mixed_dataset_train.txt"
    VAL_PAIRS_MODEL_SEL = "txt/mixed_dataset_val_choose_cross.txt"
    VAL_PAIRS_THRES_SEL = "txt/mixed_dataset_val_cross.txt"
    TEST_PAIRS = "txt/mixed_dataset_test_cross.txt"

    # AdaFace uses BGR -- should I revert conversion read_image here?

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_bias(self, bias):
        if self.biased:
            self.bias = bias % self.__len__()

    def _process_labels(self, sample):
        kin_id = self.sample_cls.NAME2LABEL[sample.kin_relation]
        race_id = RACE_DICT[sample.race]
        labels = (kin_id, sample.is_kin, race_id[0])
        return labels

    def __getitem__(self, item):
        sample = self.sample_list[(item + self.bias) % len(self)]
        img1, img2 = self._process_images(sample)
        labels = self._process_labels(sample)
        return img1, img2, labels


class KFCDataModule(L.LightningDataModule):
    DATASETS = {"kfc": FIWKFC}

    def __init__(self, dataset: str, biased: bool, batch_size=20, root_dir=".", transform=None):
        super().__init__()
        self.dataset = self.DATASETS[
            dataset
        ]  # Couldn't find a better way to do this; if I put into init, I can't make it work
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.transform = transform or T.Compose([T.ToTensor()])
        self.biased = biased

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=FIWKFC.TRAIN_PAIRS,
                batch_size=self.batch_size,
                biased=self.biased,
                transform=self.transform,
                sample_cls=SampleKFC,
            )
            self.val_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=FIWKFC.VAL_PAIRS_MODEL_SEL,
                batch_size=self.batch_size,
                transform=self.transform,
                sample_cls=SampleKFC,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=FIWKFC.VAL_PAIRS_THRES_SEL,
                batch_size=self.batch_size,
                transform=self.transform,
                sample_cls=SampleKFC,
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset(
                root_dir=self.root_dir,
                sample_path=FIWKFC.TEST_PAIRS,
                batch_size=self.batch_size,
                transform=self.transform,
                sample_cls=SampleKFC,
            )
        print(f"Setup {stage} datasets")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)


if __name__ == "__main__":
    datamodule = KFCDataModule(dataset="kfc", biased=True, batch_size=20, root_dir="../datasets/kfc")
    datamodule.setup("fit")
    train_batch = next(iter(datamodule.train_dataloader()))
    val_model_batch = next(iter(datamodule.val_dataloader()))
    datamodule.setup("validate")
    val_thres_batch = next(iter(datamodule.val_dataloader()))
    datamodule.setup("test")
    test_batch = next(iter(datamodule.test_dataloader()))
