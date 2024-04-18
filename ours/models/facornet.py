import numpy as np
import torch
import torch.nn as nn
import torchmetrics as tm
from datasets.utils import SampleKFC
from losses import contrastive_loss
from models.base import LightningBaseModel, load_pretrained_model
from models.utils import l2_norm
from pytorch_metric_learning.losses import ArcFaceLoss


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor


class HeadKin(nn.Module):
    def __init__(self, in_features=512, out_features=4, ratio=8):
        super().__init__()
        self.projection_head = nn.Sequential(
            # TODO: think better
            torch.nn.Linear(2 * in_features, in_features // ratio),
            torch.nn.BatchNorm1d(in_features // ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // ratio, out_features),
        )

        self.initialize_weights(self.projection_head)

    def initialize_weights(self, proj_head):
        for m in proj_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight - 0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

    def forward(self, em):
        return self.projection_head(em)


class HeadFamily(nn.Module):
    def __init__(self, in_features=512, out_features=4, ratio=2):
        super().__init__()
        self.projection_head = nn.Sequential(
            torch.nn.Linear(in_features, in_features // ratio),
            torch.nn.BatchNorm1d(in_features // ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // ratio, out_features),
        )

    def forward(self, em):
        return self.projection_head(em)


class FaCoR(torch.nn.Module):
    def __init__(self, attention: nn.Module):
        super(FaCoR, self).__init__()
        self.backbone = load_pretrained_model("ir_101")
        self.attention = attention

    def forward(self, imgs, aug=False):
        img1, img2 = imgs
        idx = [2, 1, 0]
        f1_0, x1_feat = self.backbone(img1[:, idx])  # (B, 512) and (B, 512, 7, 7)
        f2_0, x2_feat = self.backbone(img2[:, idx])  # ...

        # Both are (B, 512)
        f1_0 = l2_norm(f1_0)
        f2_0 = l2_norm(f2_0)

        f1s, f2s, attention_map = self.attention(f1_0, x1_feat, f2_0, x2_feat)

        return f1s, f2s, attention_map


class FaCoRV2(FaCoR):

    def forward(self, imgs, aug=False):
        img1, img2 = imgs
        idx = [2, 1, 0]
        f1_0, x1_feat = self.backbone(img1[:, idx])  # (B, 512) and (B, 512, 7, 7)
        f2_0, x2_feat = self.backbone(img2[:, idx])  # ...

        # Both are (B, 512)
        f1_0 = l2_norm(f1_0)
        f2_0 = l2_norm(f2_0)

        # Both are (B, 512, 7, 7)
        x1_feat = l2_norm(x1_feat)
        x2_feat = l2_norm(x2_feat)

        # (B, 512), (B, 512), (B, 49, 49)
        f1s, f2s, attention_map = self.attention(f1_0, x1_feat, f2_0, x2_feat)

        return f1s, f2s, attention_map


# Define a custom L2 normalization layer
class L2Norm(nn.Module):
    def __init__(self, axis=1):
        super(L2Norm, self).__init__()
        self.axis = axis

    def forward(self, x):
        # L2 normalization
        return nn.functional.normalize(x, p=2, dim=self.axis)


class FaCoRNetLightning(LightningBaseModel):

    def _step(self, inputs):
        f1, f2, att = self(inputs)
        loss = self.criterion(f1, f2, beta=att)
        sim = torch.cosine_similarity(f1, f2)
        psi = self.criterion.m(att)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2, att], "psi": psi}
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, _ = batch
        outputs = self._step([img1, img2])
        loss = outputs["contrastive_loss"]
        psi = outputs["psi"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_histogram("psi/train", psi, self.global_step)

        return loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step([img1, img2])
        psi = outputs["psi"]
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)
        self.logger.experiment.add_histogram("psi/val", psi, self.global_step)


class FaCoRNetLightningV2(LightningBaseModel):

    def _step(self, inputs):
        f1, f2, att = self(inputs)
        loss = self.criterion(f1, f2, beta=att)
        sim = torch.cosine_similarity(f1, f2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2, att]}
        # debug temperature (improve it some callback)
        psi = self.criterion.m(att)
        outputs["psi_regularization"] = 0.01 * psi.mean()
        if self.training:
            self.logger.experiment.add_histogram("psi/train", psi, self.global_step)
        else:
            self.logger.experiment.add_histogram("psi/val", psi, self.global_step)
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, _ = batch
        outputs = self._step([img1, img2])
        contrastive_loss = outputs["contrastive_loss"]
        total_loss = contrastive_loss - outputs["psi_regularization"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step([img1, img2])
        total_loss = outputs["contrastive_loss"] - outputs["psi_regularization"]
        self.log(f"loss/{stage}", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FaCoRNetMTFamily(FaCoRNetLightning):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nf = self.hparams.num_families
        self.classifier = HeadFamily(in_features=512, out_features=self.nf)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy_family = tm.Accuracy(
            num_classes=self.nf, task="multiclass"
        )  # nf will change for val and test, therefore won't be compute

    def _get_logits(self, x):
        if isinstance(self.loss, ArcFaceLoss):
            logits = self.loss.get_logits(x)
        else:
            logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):
        pair_batch, family_batch = batch
        img1, img2, labels = pair_batch
        imgs, families, _ = family_batch
        # Forward pass
        outputs = super()._step([img1, img2])
        fam_features, _ = self.model.backbone(imgs[:, [2, 1, 0]])  # why? the original code has this.
        fam_features = l2_norm(fam_features)
        fam_logits = self._get_logits(fam_features)
        # Compute losses
        contrastive_loss = outputs["contrastive_loss"]
        family_loss = self.loss(fam_logits, families)
        if self.hparams.loss_factor:
            loss = (1 - self.hparams.loss_factor) * contrastive_loss + self.hparams.loss_factor * family_loss
        else:
            loss = contrastive_loss + family_loss
        # Compute and log family accuracy
        family_accuracy = self.accuracy_family(fam_logits, families)
        self.log(
            "accuracy/classification/family", family_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        # Log lr and losses
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/contrastive/train", contrastive_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/classification/train", family_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        # Forward pass
        outputs = super()._step([img1, img2])
        # Compute losses
        contrastive_loss = outputs["contrastive_loss"]
        # Log losses
        self.log(f"loss/{stage}", contrastive_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FamilyClassifier(FaCoRNetLightning):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = load_pretrained_model("ir_101")
        self.nf = self.hparams.num_families
        self.classifier = HeadFamily(in_features=512, out_features=self.nf)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy_family = tm.Accuracy(
            num_classes=self.nf, task="multiclass"
        )  # nf will change for val and test, therefore won't be compute

    def forward(self, inputs):
        return self.model(inputs[:, [2, 1, 0]])[0]

    def training_step(self, batch, batch_idx):
        imgs, families, _ = batch
        # Forward pass
        fam_features = self(imgs)
        fam_features = l2_norm(fam_features)
        fam_preds = self.classifier(fam_features)
        # Compute losses
        loss = self.cross_entropy(fam_preds, families)
        # Compute and log family accuracy
        family_accuracy = self.accuracy_family(fam_preds, families)
        self.log(
            "accuracy/classification/family", family_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # Log lr and losses
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        # Forward pass
        f1 = self(img1)
        f2 = self(img2)
        sim = torch.cosine_similarity(f1, f2)
        # Compute losses
        loss = contrastive_loss(f1, f2, beta=0.08)  # R2021
        # Log losses
        self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(sim)
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FaCoRNetKinRace(FaCoRNetLightning):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_cls = SampleKFC

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin, _ = labels
        outputs = self._step([img1, img2])
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


if __name__ == "__main__":
    model = FaCoRV2()
    model.eval()
    img = torch.randn(2, 3, 112, 112)
    model((img, img))
