from argparse import ArgumentParser
from functools import singledispatchmethod

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torchmetrics as tm
from pytorch_metric_learning import losses
from sklearn.model_selection import KFold
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from config import LOGGER
from insight_face.recognition.arcface_torch.backbones import get_model
from lr_scheduler import PolyScheduler
from utils import log_results


class Model(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 570,  # FIW train families
        embedding_dim: int = 512,
        normalize: bool = False,
        lr: float = 1e-4,
        start_lr: float = 1e-10,
        end_lr: float = 1e-10,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        scheduler: str = "multistep",
        lr_steps: tuple = (8, 14, 25, 35, 40, 50, 60),
        lr_factor: float = 0.75,
        warmup: int = 200,
        cooldown: int = 400,
        loss: str = "ce",
        model: str = "resnet101",
        arcface_m: float = 0.5,
        arcface_s: float = 64,
        weights: str = "",
        insightface: bool = False,
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

        self.lr = lr
        self.start_lr = start_lr
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

        self.weights = weights
        self.insightface = insightface
        self.model = model
        self.loss = loss

        # We need this here because of .load_from_checkpoint,
        # which needs to know the model architecture after __init__, I suppose.
        self._init_metrics()
        self._init_model()
        self._init_loss()

        self._load_model()

        self.save_hyperparameters()

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

    def _init_model(self):

        if self.insightface:
            self.backbone = get_model("r100", fp16=False)
        else:
            self.backbone = timm.models.create_model(
                self.model, num_classes=0, global_pool=""
            )
            out_features = self.backbone(torch.randn(1, 3, 112, 112))
            self._out_features = torch.flatten(out_features, 1).shape[1]
            # TODO: rename fc to embeddings/features
            self.fc = torch.nn.Linear(self._out_features, self.embedding_dim)
            self.bn = torch.nn.BatchNorm1d(
                self.embedding_dim,
                eps=1e-5,
                momentum=0.1,
                affine=True,  # default parmas
            )
            torch.nn.init.constant_(self.bn.weight, 1.0)
            self.bn.weight.requires_grad = False

        if self.loss != "arcface":
            # Classification layer
            # TODO: temporary name
            self.classification = torch.nn.Linear(
                self.embedding_dim, self.num_classes
            )  # init?
            torch.nn.init.normal_(self.classification.weight, std=0.01)

    def _load_model(self):
        # if passed weights is not from insightface, load it
        # otherwise, it was loaded in _init_model
        if self.weights:
            # load checkpoint without insightface
            if not self.insightface:
                state_dict = torch.load(self.weights)
                if self.weights.endswith(".ckpt"):
                    state_dict = state_dict["state_dict"]
                # task=finetune doesn't load loss.W and loss.b for arcface loss
                self.load_state_dict(state_dict, strict=False)
            else:
                # load checkpoint after finetuning
                if self.weights.endswith(".ckpt"):
                    state_dict = torch.load(self.weights)["state_dict"]
                    self.load_state_dict(state_dict, strict=False)
                    print("Loaded insightface model from ckpt.")
                else:
                    # load insightface checkpoint for pretraining
                    self.backbone = get_model("r100", fp16=False)
                    self.backbone.load_state_dict(torch.load(self.weights))
                    print("Loaded insightface model.")

    def setup(self, stage):
        pass

    def configure_optimizers(self):
        if not self.warmup:
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
                    max_steps=self.trainer.estimated_stepping_batches,
                    warmup_steps=self.warmup,
                ),
                "interval": "step",
                "frequency": 1,
            }

        print(f"optimizers config = {config}")
        LOGGER.info(
            "Model will train for steps=%s", self.trainer.estimated_stepping_batches
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
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
        # cool down lr
        elif (
            self.trainer.global_step
            > self.trainer.estimated_stepping_batches - self.cooldown  # cooldown start
        ):
            cur_lr = (
                self.trainer.estimated_stepping_batches - self.trainer.global_step
            ) * (
                optimizer.param_groups[0]["lr"] - self.end_lr
            ) / self.cooldown + self.end_lr
            optimizer.param_groups[0]["lr"] = cur_lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def _forward_features(self, x):
        embeddings = self.backbone(x)

        if not self.insightface:
            embeddings = torch.flatten(embeddings, 1)
            embeddings = self.fc(embeddings)
            embeddings = self.bn(embeddings)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _get_logits(self, x):
        if isinstance(self.loss, losses.ArcFaceLoss):
            logits = self.loss.get_logits(x)
        else:
            logits = self.classification(x)
        return logits

    def forward(self, x):
        embeddings = self._forward_features(x)
        logits = self._get_logits(embeddings)
        return embeddings, logits

    def _base_step(self, batch):
        images, labels = batch[0], batch[1]
        embeddings, logits = self(images)

        if isinstance(self.loss, losses.ArcFaceLoss):
            loss = self.loss(embeddings, labels)
        else:
            loss = self.loss(logits, labels)

        preds = logits.argmax(dim=1)
        return preds, loss

    def training_step(self, batch, batch_idx):
        preds, loss = self._base_step(batch)
        acc = self.train_accuracy(preds, batch[1])
        self.log(
            "train/accuracy",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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

    def _compute_validation_dist(self, outputs):
        # TODO: works for FIW with Model. Test it for LFW and others with PretrainModel.
        embs1, embs2, labels, similarities = zip(*[batch.values() for batch in outputs])
        similarities = torch.cat(similarities, dim=0)
        labels = torch.cat(labels, dim=0)
        embs1 = torch.cat(embs1, dim=0)
        embs2 = torch.cat(embs2, dim=0)
        normed_emb1 = embs1 / torch.linalg.norm(embs1, axis=-1, keepdims=True)
        normed_emb2 = embs2 / torch.linalg.norm(embs2, axis=-1, keepdims=True)
        diff = normed_emb1 - normed_emb2
        distances = torch.linalg.norm(diff, dim=-1)

        return distances, similarities, labels

    def _log_validation_metrics(self, distances, similarities, labels):
        # TODO: change tag names for 'verification'
        # logs histograms of distances and similarities,
        # plots accuracy vs thresholds for distances and similarities
        # plots pr curve, roc, and computes and logs auc
        # at last, computs best accuracy in a KFold scheme

        best_threshold, best_accuracy, auc = log_results(
            self.logger.experiment,
            "val",
            distances.cpu().numpy(),
            similarities.cpu().numpy(),
            labels.cpu().numpy(),
            self.trainer.global_step,
            log_auc=False,
        )

        self.log(
            "val/auc",
            auc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val/accuracy",
            best_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/threshold",
            best_threshold,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_epoch_end(self, outputs):

        # TODO: is there a better way?
        # Without it, plot_roc raises an error because of logger.log_dir
        if self.trainer.fast_dev_run:
            return

        distances, similarities, labels = self._compute_validation_dist(outputs)
        self._log_validation_metrics(distances, similarities, labels)

    def test_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            preds, loss = self._base_step(batch)
            acc = self.train_accuracy(preds, batch[1])
            self.log(
                "train/accuracy",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        elif dataloader_idx == 1:
            return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        # outputs is a list of outputs for each dataloader

        # verification metrics
        distances, similarities, labels = self._compute_validation_dist(outputs[1])
        self._log_validation_metrics(distances, similarities, labels)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # These default values are specific for pretraining.
        # I think I should put them in PretrainModel.
        parser = parent_parser.add_argument_group("SiameseNet")
        parser.add_argument(
            "--num-classes",
            default=570,  # num_families in fiw train set
            type=int,
        )
        parser.add_argument(
            "--lr",
            default=1e-10,
            type=float,
        )
        parser.add_argument(
            "--start-lr",
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
            default=0.75,
            type=float,
        )
        parser.add_argument(
            "--lr-steps",
            default=(8, 14, 25, 35, 40, 50, 60),
            type=int,
            nargs="+",
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
        parser.add_argument("--scheduler", type=str, default="multistep")
        parser.add_argument("--loss", type=str, default="ce")
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

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):

        # TODO: test it
        if batch_idx % 10 != 0:
            return

        first, second, label = batch
        img1, img2 = first[0], second[0]
        images = torch.cat((img1, img2), dim=0)
        bs = len(label)

        embs1 = outputs["embs1"]
        embs2 = outputs["embs2"]
        # cat embeddings
        embs = torch.cat([embs1, embs2], dim=0)
        # create labels for all embeddings
        pair_names = np.array(["img1", "img2"]).repeat(bs)
        sample_idx_str_arr = np.tile([f"_{i}" for i in range(bs)], 2)
        labels = np.char.add(pair_names, sample_idx_str_arr)
        # TODO: add label to labels strings

        self.logger.experiment.add_embedding(
            embs,
            metadata=labels,
            label_img=images,
            global_step=self.global_step,
            tag=f"embeddings/{batch_idx}",
        )

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
        norms = torch.cat(norms, dim=0)
        dists = torch.cat(dists, dim=0)
        labels = torch.cat(labels, dim=0)

        # init vars
        n_pairs = dists.shape[0]
        accuracy = torch.zeros((n_folds), dtype=torch.float32)
        # i think i got this end=4 from insightface, which makes sense with clip_grad_val.
        thresholds = torch.arange(0.0, 4, 0.01, dtype=torch.float32)
        best_thresholds = torch.zeros((n_folds), dtype=torch.float32)
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
            best_thresholds[fold] = threshold

        acc = torch.mean(accuracy).item()
        norm = norms.mean()
        best_threshold = torch.mean(best_thresholds).item()

        positive_samples = labels == 1
        negative_samples = labels == 0

        if any(negative_samples):
            self.logger.experiment.add_histogram(
                "distances/negative samples distribution",
                dists[negative_samples],
                self.global_step,
            )
        if any(positive_samples):
            self.logger.experiment.add_histogram(
                "distances/positive samples distribution",
                dists[positive_samples],
                self.global_step,
            )

        self.log(
            f"{target}/threshold",
            best_threshold,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            f"{target}/acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{target}/norm",
            norm,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
