from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import lr_monitor

import mytypes as t
import transforms as mytransforms
from dataset import FamiliesDataset, FamiliesDataModule, MS1MDataset, MS1MDataModule
from model import Model


if __name__ == "__main__":
    seed_everything(42, workers=True)
    parser = ArgumentParser()

    # Program specific args
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=["pretrain", "finetune"]
    )
    parser.add_argument(
        "--batch-size",
        default=48,
        type=int,
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    # Model specific args
    parser = Model.add_model_specific_args(parser)

    # Trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(None)

    # TODO: add to argparse
    jitter_param = 0.15
    lighting_param = 0.15

    if args.task == "pretrain":
        train_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        val_transforms = transforms.Compose([transforms.ToTensor()])

        datamodule = MS1MDataModule(
            Path(args.data_dir),
            transforms=[train_transforms, val_transforms],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        datamodule.setup(stage="fit")
        num_classes = datamodule.train_dataloader().dataset.num_classes
        num_samples = len(datamodule.train_dataloader().dataset)

        # instantiate model
        model = Model(num_classes, num_samples, args)
        model.train()
        lrm = lr_monitor.LearningRateMonitor(logging_interval="step")

        # show params
        print(args)

        # instantiate trainer
        trainer = Trainer.from_argparse_args(args, callbacks=[lrm])
        # fit model on dataset
        trainer.fit(model, datamodule)
    else:
        train_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ColorJitter(
                    brightness=jitter_param,
                    contrast=jitter_param,
                    saturation=jitter_param,
                ),
                mytransforms.Lightning(
                    lighting_param,
                    mytransforms._IMAGENET_PCA["eigval"],
                    mytransforms._IMAGENET_PCA["eigvec"],
                ),
                # ReJPGTransform(0.3, 70),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose([transforms.ToTensor()])

        # add params from args
        datamodule = FamiliesDataModule(
            Path(args.data_dir),
            transforms=[train_transforms, val_transforms],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        # datamodule.setup(stage='fit')

        args.num_families = len(datamodule.train_dataloader().dataset.families)
        args.num_samples = len(datamodule.train_dataloader().dataset)

        # instantiate model
        model = Model(args)
        model.train()

        print(args)

        lrm = lr_monitor.LearningRateMonitor(logging_interval="step")

        # instantiate trainer
        trainer = Trainer.from_argparse_args(args, callbacks=[lrm])
        # fit model on dataset
        trainer.fit(model, datamodule)
