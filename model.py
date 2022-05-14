from argparse import ArgumentParser
from functools import singledispatchmethod

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
from sklearn.model_selection import KFold

from lr_scheduler import PolyScheduler
from utils import plot_roc


class Model(pl.LightningModule):
    # TODO: review params default for this Model
    def __init__(
        self,
        num_classes: int = 85742,
        embedding_dim: int = 512,
        normalize: bool = False,
        clip_gradient: float = 5.0,
        lr: float = 0.01,
        end_lr: float = 0.0001,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        scheduler: str = "multistep",
        lr_steps: tuple = (8, 14, 25, 35, 40, 50, 60),
        lr_factor: float = 0,
        warmup: int = 0,
        cooldown: int = 0,
        loss: str = "arcface",
        model: str = "resnet101",
        arcface_m: float = 0.5,
        arcface_s: float = 64,
        weights: str = "",
        insightface_weights: str = "",
        **kwargs,
    ):
        """

        Parameters
        ----------

        TODO

        """
        super().__init__()
        self.num_classes = num_classes

        self.normalize = normalize
        self.embedding_dim = embedding_dim
        self.clip_gradient = clip_gradient

        self.lr = lr
        self.end_lr = end_lr
        self.lr_factor = lr_factor
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.cooldown = cooldown
        self.lr_steps = lr_steps
        self.scheduler = scheduler

        self.margin = np.rad2deg(arcface_m)
        self.scale = arcface_s

        self.model = model
        self.loss = loss
        self.insightface = insightface_weights

        # We need this here because of .load_from_checkpoint,
        # which needs to know the model architecture after __init__, I suppose.
        self._init_metrics()
        self._init_model()
        self._init_loss()

        self.save_hyperparameters()

        if weights and not self.insightface:
            state_dict = torch.load(weights)
            if weights.endswith(".ckpt"):
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict)

    def _init_loss(self):
        if self.loss == "arcface":
            self.loss = losses.ArcFaceLoss(
                num_classes=self.num_classes,
                embedding_size=self.embedding_dim,
                margin=self.margin,
                scale=self.scale,
            )
        elif self.loss == "ce":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def _init_metrics(self):
        self.train_accuracy = tm.Accuracy()
        self.val_accuracy = tm.Accuracy()
        self.val_auc = tm.AUROC()
        self.val_roc = tm.ROC()

    def _init_model(self):
        if self.insightface:
            from insight_face.recognition.arcface_torch.backbones import get_model

            self.backbone = get_model(self.model, fp16=False)
            self.backbone.load_state_dict(torch.load(self.insightface))
            print("Loaded insightface model.")
        else:
            self.backbone = timm.models.create_model(
                self.model, num_classes=0, global_pool=""
            )
            out_features = self.backbone(torch.randn(1, 3, 112, 112))
            out_features = torch.flatten(out_features, 1).shape[1]
            self.fc = torch.nn.Linear(out_features, self.embedding_dim)
            self.bn = torch.nn.BatchNorm1d(
                self.embedding_dim,
                eps=1e-5,
                momentum=0.1,
                affine=True,  # default parmas
            )
            torch.nn.init.constant_(self.bn.weight, 1.0)
            self.bn.weight.requires_grad = False

        # Will I use another loss?
        if self.loss != "arcface":
            self.fc = torch.nn.Linear(self.embedding_dim, self.num_classes)

    def setup(self, stage):
        # TODO: dont need to call super().setup?
        if stage == "fit":
            # for cooldown lr
            # src: https://github.com/PyTorchLightning/pytorch-lightning/issues/3115#issuecomment-678824664
            # 1.5.11 will include this in trainer, I think. My version is 1.5.10
            if isinstance(self.trainer.gpus, int):
                gpus = 1 if self.trainer.gpus else 0
                if self.trainer.gpus == -1:
                    gpus = torch.cuda.device_count()
            else:
                # TODO: improve-me
                gpus = len(self.trainer.gpus.split(","))
            total_devices = gpus * self.trainer.num_nodes
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
        elif stage == "validate":
            pass
        else:
            pass

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
            config["lr_scheduler"] = {
                "scheduler": PolyScheduler(
                    optimizer,
                    base_lr=self.lr,
                    max_steps=self.train_steps,
                    warmup_steps=self.warmup,
                ),
                "interval": "step",
                "frequency": 1,
            }

        print(f"optimizers config = {config}")
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
        if self.insightface:
            return self.backbone(x)
        embeddings = self.backbone(x)
        embeddings = torch.flatten(embeddings, 1)
        embeddings = self.fc(embeddings)
        embeddings = self.bn(embeddings)
        if self.normalize:
            # For ArcFaceLoss we normalize the embeddings and weights internally.
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _get_logits(self, x):
        if isinstance(self.loss, losses.ArcFaceLoss):
            logits = self.loss.get_logits(x)
        else:
            logits = self.fc(x)
        return logits

    def forward(self, x):
        embeddings = self._forward_features(x)
        logits = self._get_logits(embeddings)
        return embeddings, logits

    def __base_step(self, batch):
        images, labels = batch[0], batch[1]
        embeddings, logits = self(images)

        if isinstance(self.loss, losses.ArcFaceLoss):
            loss = self.loss(embeddings, labels)
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
        img1, img2, label = batch
        embeddings1, _ = self(img1)
        embeddings2, _ = self(img2)
        similarity = F.cosine_similarity(embeddings1, embeddings2, dim=-1)

        return {
            "embs1": embeddings1,
            "embs2": embeddings2,
            "label": label,
            "similarity": similarity,
        }

    def validation_epoch_end(self, outputs):

        # TODO: works for FIW with Model. Test it for LFW and others with PretrainModel.
        _, _, labels, similarities = zip(*[batch.values() for batch in outputs])
        similarities = torch.cat(similarities, dim=0)
        labels = torch.cat(labels, dim=0)

        preds = (
            similarities > 0.5
        )  # add as attribute and update at each validation_epoch_end?
        acc = self.val_accuracy(preds, labels)
        fpr, tpr, _ = self.val_roc(similarities, labels)
        fig = plot_roc(tpr.cpu(), fpr.cpu(), savedir=self.logger.log_dir)
        auc = self.val_auc(similarities, labels)

        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.logger.experiment.add_figure("ROC Curve", fig, self.current_epoch)
        self.log(
            "val_auc",
            auc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        # These default values are specific for pretraining.
        # I think I should put them in PretrainModel.
        parser = parent_parser.add_argument_group("SiameseNet")
        parser.add_argument(
            "--num-classes",
            default=85742,
            type=int,
        )
        parser.add_argument(
            "--lr",
            default=0.1,
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
            default=5e-4,
            type=float,
        )
        parser.add_argument(
            "--clip-gradient",
            default=1.5,
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
        parser.add_argument("--model", default="resnet101", type=str)
        # TODO: use this flag for insightface?
        parser.add_argument(
            "--weights", help="Pretrained weights.", default="", type=str
        )
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
        parser.add_argument("--scheduler", type=str, default="poly")
        parser.add_argument("--loss", type=str, default="arcface")
        return parent_parser


class PretrainModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage: str):
        super().setup(stage)
        if stage == "test":
            self.val_targets = ["cfp_fp", "agedb_30"]
        else:
            self.val_targets = ["lfw"]

    # TODO: use singledispatchmethod to overload validation_step

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, self.val_targets[0])

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self._step(batch, batch_idx, dataloader_idx)

    def test_epoch_end(self, outputs):
        for idx, output in enumerate(outputs):
            self._epoch_end(output, self.val_targets[idx])

    def _step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        first, second, label = batch

        first_emb = self._forward_features(first[0])
        first_flipped_emb = self._forward_features(first[1])
        embeddings1 = first_emb + first_flipped_emb
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)

        second_emb = self._forward_features(second[0])
        second_flipped_emb = self._forward_features(second[1])
        embeddings2 = second_emb + second_flipped_emb
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        diff = embeddings1 - embeddings2
        dist = torch.linalg.norm(diff, dim=-1)

        embeddings = torch.cat(
            [first_emb, first_flipped_emb, second_emb, second_flipped_emb], dim=0
        )
        norm = torch.linalg.norm(embeddings, dim=-1)
        norm = torch.mean(norm, dim=0, keepdim=True)  # to make norm.dim = 1

        return {
            "embs1": first_emb,
            "embs2": second_emb,
            "norm": norm,
            "dist": dist,
            "label": label,
        }

    def _epoch_end(self, outputs, target):

        n_folds = 10
        kfold = KFold(n_splits=n_folds, shuffle=False)

        # unpack output in norms, dists, labels from output list
        _, _, norms, dists, labels = zip(*[batch.values() for batch in outputs])

        # stack batches alonge batch_dim
        norm = torch.cat(norms, dim=0).mean()
        dists = torch.cat(dists, dim=0)
        labels = torch.cat(labels, dim=0)

        # init vars
        n_pairs = dists.shape[0]
        accuracy = torch.zeros((n_folds), dtype=torch.float32)
        thresholds = torch.arange(0.0, 4, 0.01, dtype=torch.float32)
        indexes = torch.arange(n_pairs, dtype=torch.int32)
        acc_train = torch.zeros((thresholds.shape[0]), dtype=torch.float32)

        # iterate over folds
        for fold, (train_idx, test_idx) in enumerate(kfold.split(indexes)):
            # slice train dist and labels
            train_dists = dists[train_idx]
            train_labels = labels[train_idx]
            # compute train accuracy
            for t_idx, threshold in enumerate(thresholds):
                predicted = train_dists < threshold
                acc_train[t_idx] = tm.functional.accuracy(predicted, train_labels)
            # compute test accuracy
            test_dists = dists[test_idx]
            test_labels = labels[test_idx]
            best_threshold_index = torch.argmax(acc_train)
            threshold = thresholds[best_threshold_index]
            predicted = test_dists < threshold
            accuracy[fold] = tm.functional.accuracy(predicted, test_labels)

        acc = torch.mean(accuracy).item()

        self.log(
            f"{target}_acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{target}_norm",
            norm,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
