from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from pl_bolts.callbacks import ModuleDataMonitor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import lr_monitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch import nn, utils
from torchvision import transforms, datasets
from tqdm import tqdm

import mytypes as t
from callbacks import ModelInspectionCallback, MetricsCallback
from model import Model, PretrainModel
from tasks import init_cifar, init_ms1m, init_fiw


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


if __name__ == "__main__":
    seed_everything(42, workers=True)

    # TODO: add to argparse

    parser = init_parser()
    args = parser.parse_args(None)

    trainer = Trainer.from_argparse_args(
        args,
        devices=1,
        accelerator="gpu",
    )

    if args.task == "cifar":
        raise NotImplementedError("CIFAR validation not implemented.")
    elif args.task == "pretrain":
        # instantiate model
        model = PretrainModel(args)
        model.eval()
        print(model)
        datamodule = init_ms1m(args)
        if not args.test:
            trainer.validate(model, datamodule, ckpt_path=args.ckpt_path)
        else:
            trainer.test(model, datamodule, ckpt_path=args.ckpt_path)
    elif args.task == "finetune":
        raise NotImplementedError("Finetune validation not implemented yet.")
