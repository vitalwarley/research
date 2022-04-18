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
    parser.add_argument(
        "--task", type=str, required=True, choices=["cifar", "pretrain", "finetune"]
    )
    parser.add_argument(
        "--batch-size",
        default=48,
        type=int,
    )
    parser.add_argument(
        "--num-samples",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
    )
    parser.add_argument("--debug", type=str, default="", nargs="+")

    # Model specific args
    parser = Model.add_model_specific_args(parser)

    # Trainer args
    parser = Trainer.add_argparse_args(parser)

    return parser


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


if __name__ == "__main__":
    seed_everything(42, workers=True)

    # TODO: add to argparse

    parser = init_parser()
    args = parser.parse_args(None)
    trainer = init_trainer(args)

    if args.task == "cifar":
        # instantiate model
        model = Model(args)
        model.train()

        print(model)

        trainloader, valloader = init_cifar(args)
        model.train_dataloader = trainloader
        trainer.fit(model, trainloader, valloader)
    elif args.task == "pretrain":
        # instantiate model
        model = PretrainModel(args)
        model.train()

        print(model)

        datamodule = init_ms1m(args)
        trainer.fit(model, datamodule)
    elif args.task == "finetune":
        # instantiate model
        model = Model(args)
        model.train()

        print(model)

        datamodule = init_fiw(args)
        trainer.fit(model, datamodule)
