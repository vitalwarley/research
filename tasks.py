from argparse import ArgumentParser

from torch import utils
from torchvision import transforms, datasets
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from pl_bolts.callbacks import ModuleDataMonitor

import transforms as mytransforms
from callbacks import ModelInspectionCallback, MetricsCallback
from dataset import FamiliesDataset, FamiliesDataModule, MS1MDataModule
from model import Model


def init_trainer(args):

    # add callbacks
    # lrm_cb = lr_monitor.LearningRateMonitor(logging_interval="step")
    mdm_cb = ModuleDataMonitor(submodules=True, log_every_n_steps=1)
    mi_cb = ModelInspectionCallback()
    m_cb = MetricsCallback()

    # callbacks = [lrm_cb]
    # callbacks = [m_cb]
    callbacks = []

    if "monitoring" in args.debug:
        callbacks.append(mdm_cb)

    if "weights" in args.debug:
        callbacks.append(mi_cb)

    # show params
    print(args)

    # instantiate trainer
    trainer = Trainer.from_argparse_args(
        args, callbacks=callbacks, strategy=DDPStrategy(find_unused_parameters=False), accelerator='gpu'
    )

    return trainer


def init_parser():
    parser = ArgumentParser()

    # Program specific args
    parser.add_argument(
        "--data-dir",
        type=str,
    )
    # parser.add_argument("--insightface", action='store_true')
    parser.add_argument("--ckpt-path", type=str)  # ptl ckpt
    parser.add_argument("--insightface-weights", type=str)  # insightface
    parser.add_argument(
        "--task", type=str, required=True, choices=["cifar", "pretrain", "finetune"]
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
        "--num-samples",
        default=0,
        type=int,
    )
    parser.add_argument("--debug", type=str, default="", nargs="+")
    parser.add_argument("--test", action="store_true")

    # Model specific args
    parser = Model.add_model_specific_args(parser)

    # Trainer args
    parser = Trainer.add_argparse_args(parser)

    return parser


def init_cifar(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = datasets.CIFAR10(
        root="~/datasets", train=True, download=True, transform=transform
    )
    trainloader = utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    valset = datasets.CIFAR10(
        root="~/datasets", train=False, download=True, transform=transform
    )
    valloader = utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainloader, valloader


def init_ms1m(args):
    train_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    datamodule = MS1MDataModule(
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        data_dir=args.data_dir,
        transforms=[train_transforms, val_transforms],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return datamodule


def init_fiw(args):
    train_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ColorJitter(
                brightness=args.jitter_param,
                contrast=args.jitter_param,
                saturation=args.jitter_param,
            ),
            mytransforms.Lightning(
                args.lighting_param,
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
        args.data_dir,
        transforms=[train_transforms, val_transforms],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return datamodule
