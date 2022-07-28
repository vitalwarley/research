from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from pl_bolts.callbacks import ModuleDataMonitor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import lr_monitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch import nn, utils
from torchvision import datasets, transforms
from tqdm import tqdm

import mytypes as t
from callbacks import ModelInspectionCallback
from model import Model, PretrainModel
from tasks import init_cifar, init_fiw, init_ms1m, init_parser

if __name__ == "__main__":
    seed_everything(42, workers=True)

    # TODO: add to argparse

    parser = init_parser()
    args = parser.parse_args(None)

    trainer = Trainer.from_argparse_args(
        args,
        devices=1,
        accelerator="gpu",
        # callbacks=[ModelInspectionCallback()]
    )

    print(args)

    if args.task == "cifar":
        raise NotImplementedError("CIFAR validation not implemented.")
    elif args.task == "pretrain":
        # instantiate model
        model = PretrainModel(**vars(args))
        model.eval()
        print(model)
        datamodule = init_ms1m(**vars(args))
    elif args.task == "finetune":
        args.model = "resnet101" if not args.model else args.model
        model = Model(**vars(args))
        model.eval()
        print(model)
        datamodule = init_fiw(**vars(args))

    if not args.test:
        trainer.validate(model, datamodule, ckpt_path=args.ckpt_path)
    else:
        trainer.test(model, datamodule, ckpt_path=args.ckpt_path)
