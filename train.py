from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import lr_monitor
from torch import nn, utils
from torchvision import datasets, transforms
from tqdm import tqdm

import mytypes as t
from model import Model, PretrainModel
from tasks import init_cifar, init_fiw, init_ms1m, init_parser, init_trainer

if __name__ == "__main__":
    seed_everything(42, workers=True)

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
        model = PretrainModel(**vars(args))
        model.train()

        print(model)

        datamodule = init_ms1m(**vars(args))
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
    elif args.task == "finetune":
        # instantiate model
        model = Model(**vars(args))
        model.train()

        print(model)

        datamodule = init_fiw(**vars(args))
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
