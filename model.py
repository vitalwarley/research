from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torchmetrics as tm
from pytorch_metric_learning import losses
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from lr_scheduler import PolyScheduler


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

        self.margin = np.rad2deg(args.arcface_m)
        self.scale = args.arcface_s

        # timm automatically replaces final layer with a new, untrained, linear layer
        # we do this to normalize embeddings as in original insightface implementation (?)
        self.backbone = timm.models.create_model(
            args.model, pretrained=True, num_classes=0
        )
        # FIXME: improve
        out_features = self.backbone(torch.randn(1, 3, 112, 112)).shape[1]
        self.linear = torch.nn.Linear(out_features, self.embedding_dim)
        self.bn = torch.nn.BatchNorm1d(
            self.embedding_dim, eps=0.001, momentum=0.1, affine=False
        )

        self.loss = losses.ArcFaceLoss(
            num_classes=self.num_classes,
            embedding_size=self.embedding_dim,
            margin=self.margin,
            scale=self.scale,
        )

        self.train_accuracy = tm.Accuracy()
        self.val_accuracy = tm.Accuracy()

        self.args = args

    def setup(self, stage):
        # TODO: dont need to call super().setup?
        if stage == "fit" or stage is None:
            # for cooldown lr
            # src: https://github.com/PyTorchLightning/pytorch-lightning/issues/3115#issuecomment-678824664
            # 1.5.11 will include this in trainer, I think. My version is 1.5.10
            total_devices = self.trainer.gpus * self.trainer.num_nodes
            total_devices = total_devices if total_devices else 1
            # train_batches = len(self.train_dataloader()) // total_devices
            train_samples = len(self.trainer.datamodule.train_dataloader())
            train_batches = train_samples / self.args.batch_size // total_devices
            self.train_steps = (
                self.trainer.max_epochs * train_batches
            ) // self.trainer.accumulate_grad_batches

            print(f"train steps = {self.train_steps}")

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

        # FIXME: improve
        if self.lr_factor:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": MultiStepLR(
                        optimizer, milestones=self.lr_steps, gamma=self.lr_factor
                    )
                },
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": PolyScheduler(
                        optimizer,
                        base_lr=self.lr,
                        max_steps=self.train_steps,
                        warmup_steps=self.warmup,
                    ),
                    "interval": "step",
                    "frequency": 1,
                },
            }

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
                optimizer.param_groups["lr"] - self.end_lr
            ) / self.cooldown + self.end_lr
            optimizer.param_groups["lr"] = cur_lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def forward(self, x):
        embeddings = self.backbone(x)
        embeddings = self.linear(embeddings)
        embeddings = self.bn(embeddings)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        # if torch.any(torch.isnan(embeddings)):
        #     breakpoint()
        return embeddings

    def _calculate_loss(self, batch):
        images, labels = batch[0], batch[1]  # labels == family_idx
        logits = self(images)
        # # TODO: add clip_grad
        loss = self.loss(logits, labels)
        return loss, logits

    def __base_step(self, batch):
        loss, logits = self._calculate_loss(batch)
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
            "--num_classes",
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
        return parent_parser
