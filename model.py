from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torchmetrics as tm
from torch import nn
from pytorch_metric_learning import losses
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from lr_scheduler import PolyScheduler


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(400, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        # self.fc3 is self.fc in model

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Model(pl.LightningModule):
    def __init__(self, args: ArgumentParser = None):
        """

        Parameters
        ----------

        TODO

        """
        super().__init__()
        self.num_classes = args.num_classes

        self.normalize = args.normalize
        self.embedding_dim = args.embedding_dim
        self.clip_gradient = args.clip_gradient

        self.lr = args.lr
        self.end_lr = args.end_lr
        self.lr_factor = args.lr_factor
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.warmup = args.warmup
        self.cooldown = args.cooldown
        self.lr_steps = [8, 14, 25, 35, 40, 50, 60]
        self.scheduler = args.scheduler

        self.margin = np.rad2deg(args.arcface_m)
        self.scale = args.arcface_s

        self.train_accuracy = tm.Accuracy()
        self.val_accuracy = tm.Accuracy()

        self._init_model(args.model)

        if args.loss == "arcface":
            self.loss = losses.ArcFaceLoss(
                num_classes=self.num_classes,
                embedding_size=self.embedding_dim,
                margin=self.margin,
                scale=self.scale,
            )
        elif args.loss == "ce":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def _init_model(self, model):
        if model == "simple":
            self.backbone = SimpleModel()
            self.fc = torch.nn.Linear(84, 10)
        else:
            self.backbone = timm.models.create_model(model, num_classes=0)
            # FIXME: improve; resnet34 out_features is already 512
            out_features = self.backbone(torch.randn(1, 3, 112, 112)).shape[1]
            if out_features > 512:
                self.linear = torch.nn.Linear(out_features, self.embedding_dim)
            self.fc = torch.nn.Linear(self.embedding_dim, self.num_classes)
            # self.bn = torch.nn.BatchNorm1d(
            #     self.embedding_dim, eps=0.001, momentum=0.1, affine=False
            # )

    def setup(self, stage):

        # TODO: dont need to call super().setup?
        if stage == "fit" or stage is None:
            # for cooldown lr
            # src: https://github.com/PyTorchLightning/pytorch-lightning/issues/3115#issuecomment-678824664
            # 1.5.11 will include this in trainer, I think. My version is 1.5.10
            total_devices = self.trainer.gpus * self.trainer.num_nodes
            total_devices = total_devices if total_devices else 1
            # train_batches = len(self.train_dataloader()) // total_devices
            if self.trainer.datamodule is not None:
                train_batches = (
                    len(self.trainer.datamodule.train_dataloader()) // total_devices
                )
            else:
                train_batches = len(self.train_dataloader) // total_devices

            self.train_steps = (
                self.trainer.max_epochs * train_batches
            ) // self.trainer.accumulate_grad_batches

            print(
                f"train steps = {self.train_steps} in {train_batches} batches per epoch"
            )

    def configure_optimizers(self):
        if self.warmup > 0:
            self.start_lr = 1e-10
        else:
            self.start_lr = self.lr

        optimizer = SGD(
            # loss params are already included (ref. ArcFaceLoss)
            self.parameters(),
            lr=self.start_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        config = {
            "optimizer": optimizer,
        }

        # FIXME: improve
        if self.scheduler == "multistep":
            config["lr_scheduler"] = {
                "scheduler": MultiStepLR(
                    optimizer, milestones=self.lr_steps, gamma=self.lr_factor
                )
            }
        elif self.scheduler == "poly":
            config["lr_scheduler"] = (
                {
                    "scheduler": PolyScheduler(
                        optimizer,
                        base_lr=self.lr,
                        max_steps=self.train_steps,
                        warmup_steps=self.warmup,
                    ),
                    "interval": "step",
                    "frequency": 1,
                },
            )

        return config

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # warm up lr
        if self.trainer.global_step < self.warmup:
            cur_lr = (self.trainer.global_step + 1) * (
                self.lr - self.start_lr
            ) / self.warmup + self.start_lr
            # lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup)
            for pg in optimizer.param_groups:
                # pg["lr"] = lr_scale * self.hparams.learning_rate
                pg["lr"] = cur_lr
        # cool down lr
        elif self.trainer.global_step > self.train_steps - self.cooldown:
            cur_lr = (self.train_steps - self.trainer.global_step) * (
                optimizer.param_groups[0]["lr"] - self.end_lr
            ) / self.cooldown + self.end_lr
            optimizer.param_groups[0]["lr"] = cur_lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def _forward_features(self, x):
        embeddings = self.backbone(x)
        if hasattr(self, "linear"):
            embeddings = F.relu(self.linear(embeddings))
        if hasattr(self, "bn"):
            embeddings = self.bn(embeddings)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def forward(self, x):
        embeddings = self._forward_features(x)
        logits = self.fc(embeddings)
        return embeddings, logits

    def __base_step(self, batch):
        images, labels = batch[0], batch[1]
        embeddings, logits = self(images)

        if isinstance(self.loss, losses.ArcFaceLoss):
            loss = self.loss(embeddings, labels)
            logits = self.loss.get_logits(embeddings)
        else:
            loss = self.loss(logits, labels)

        preds = logits.argmax(dim=1)
        return preds, loss

    def training_step(self, batch, batch_idx):
        preds, loss = self.__base_step(batch)
        acc = self.train_accuracy(preds, batch[1])
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss = self.__base_step(batch)
        acc = self.val_accuracy(preds, batch[1])
        self.log(
            "val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SiameseNet")
        parser.add_argument(
            "--num-classes",
            default=93430,
            type=int,
        )
        parser.add_argument(
            "--lr",
            default=1e-4,
            type=float,
        )
        parser.add_argument(
            "--end-lr",
            default=1e-10,
            type=float,
        )
        parser.add_argument(
            "--lr-factor",
            default=0,
            type=float,
        )
        parser.add_argument(
            "--momentum",
            default=0.9,
            type=float,
        )
        parser.add_argument(
            "--weight-decay",
            default=1e-4,
            type=float,
        )
        parser.add_argument(
            "--clip-gradient",
            default=1.4,
            type=float,
        )
        parser.add_argument(
            "--warmup",
            default=200,
            type=int,
        )
        parser.add_argument(
            "--cooldown",
            default=400,
            type=int,
        )
        parser.add_argument(
            "--embedding-dim",
            default=512,
            type=int,
        )
        parser.add_argument(
            "--arcface-s",
            default=64,
            type=int,
        )
        parser.add_argument(
            "--arcface-m",
            default=0.5,
            type=float,
        )
        parser.add_argument("--model", default="vit_small_patch16_224", type=str)
        parser.add_argument(
            "--normalize",
            action="store_true",
        )
        parser.add_argument(
            "--jitter_param",
            default=0.15,
            type=float,
        )
        parser.add_argument(
            "--lighting_param",
            default=0.15,
            type=float,
        )
        parser.add_argument("--scheduler", type=str)
        parser.add_argument("--loss", type=str, default="arcface")
        return parent_parser
