import torch
import torch.nn as nn
import torchmetrics as tm
from models.base import LightningBaseModel, load_pretrained_model


class SCLFFStageOne(LightningBaseModel):
    """
    Implements "Supervised Contrastive Learning and Feature Fusion for Improved Kinship Verification".

    Stage one is contrastive learning with a Siamese network.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _step(self, inputs):
        img1, img2, labels = inputs
        f1, f2 = self((img1, img2))
        loss = self.criterion(f1, f2)
        # Cross-entropy loss based on similarities and best threshold
        sim = torch.cosine_similarity(f1, f2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2]}
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, labels = batch
        _, is_kin = labels
        outputs = self._step((img1, img2, is_kin))
        con_loss = outputs["contrastive_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", con_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return con_loss


class SCLFFStageTwo(LightningBaseModel):
    """
    Implements "Supervised Contrastive Learning and Feature Fusion for Improved Kinship Verification".

    Stage one is contrastive learning with a Siamese network.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # add loss, add datamodule, add guild/lighnintg config

    def _step(self, inputs):
        img1, img2, labels = inputs
        f1, f2 = self((img1, img2))
        loss = self.criterion(f1, f2)
        # Cross-entropy loss based on similarities and best threshold
        sim = torch.cosine_similarity(f1, f2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2]}
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, labels = batch
        _, is_kin = labels
        outputs = self._step((img1, img2, is_kin))
        con_loss = outputs["contrastive_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", con_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return con_loss
