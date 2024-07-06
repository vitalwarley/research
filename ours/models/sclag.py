import torch
import torchmetrics as tm
from models.base import LightningBaseModel
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


class SCLAG(LightningBaseModel):
    """
    Supervised Contrastive Loss with Age and Gender information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _step(self, batch, stage: str):
        img1, img2, labels = batch
        f1, f2 = self([img1, img2])
        sim = torch.cosine_similarity(f1, f2)
        features = torch.cat([f1, f2], dim=0)
        n_samples = f1.size(0)
        is_kin = labels[-1]
        positive_pairs = torch.tensor([(i, i + n_samples) for i in range(n_samples) if is_kin[i]])
        if stage == "train":
            img1_ages, img2_ages, img1_genders, img2_genders, is_kin = labels
            ages = torch.cat([img1_ages, img2_ages], dim=0)
            genders = torch.cat([img1_genders, img2_genders], dim=0)
            losses = self.criterion(features, positive_pairs, genders, ages)
            total_loss, contrastive_loss, gender_loss, age_loss = losses
            outputs = {
                "sim": sim,
                "features": [f1, f2],
                "total_loss": total_loss,
                "contrastive_loss": contrastive_loss,
                "gender_loss": gender_loss,
                "age_loss": age_loss,
            }
            return outputs
        else:
            loss = self.criterion.contrastive_loss(features, positive_pairs)
            outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2]}
            return outputs

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch, "train")
        total_loss = outputs["total_loss"]
        con_loss = outputs["contrastive_loss"]
        gender_loss = outputs["gender_loss"]
        age_loss = outputs["age_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", con_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/total", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/gender", gender_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/age", age_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return con_loss

    def _eval_step(self, batch, batch_idx, stage):
        _, _, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step(batch, stage)
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class AdversarialNetwork(nn.Module):
    def __init__(self, feature_dim=512):
        super(AdversarialNetwork, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 3)  # Gender classifier
        self.fc3 = nn.Linear(256, 4)  # Age group classifier

    def forward(self, x):
        x = F.relu(self.fc1(x))
        gender_logits = self.fc2(x)
        age_logits = self.fc3(x)
        return gender_logits, age_logits


# Reverse gradient for adversarial training
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SCLAGAdv(LightningBaseModel):
    """
    Supervised Contrastive Loss with Age and Gender information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adversarial = AdversarialNetwork(feature_dim=512)
        self.reverse = True
        self.age_acc = tm.Accuracy(task="multiclass", num_classes=4)
        self.gender_acc = tm.Accuracy(task="multiclass", num_classes=3)

    def adversarial_loss(self, features, gender_labels, age_labels):
        gender_logits, age_logits = self.adversarial(features)
        gender_loss = F.cross_entropy(gender_logits, gender_labels)
        age_loss = F.cross_entropy(age_logits, age_labels)
        return gender_logits, gender_loss, age_logits, age_loss

    def _step(self, batch, stage: str):
        img1, img2, labels = batch
        f1, f2 = self([img1, img2])
        sim = torch.cosine_similarity(f1, f2)
        features = torch.cat([f1, f2], dim=0)
        n_samples = f1.size(0)
        is_kin = labels[-1]
        positive_pairs = torch.tensor([(i, i + n_samples) for i in range(n_samples) if is_kin[i]])
        contrastive_loss = self.criterion(features, positive_pairs)
        if stage == "train":
            img1_ages, img2_ages, img1_genders, img2_genders, is_kin = labels
            ages = torch.cat([img1_ages, img2_ages], dim=0)
            genders = torch.cat([img1_genders, img2_genders], dim=0)
            if self.reverse:
                reversed_features = ReverseLayerF.apply(features, -1.0)
                gender_logits, gender_loss, age_logits, age_loss = self.adversarial_loss(
                    reversed_features, genders, ages
                )
            else:
                gender_logits, gender_loss, age_logits, age_loss = self.adversarial_loss(features, genders, ages)
            # gender_acc = tm.functional.accuracy(gender_logits, genders, task="multiclass", )
            # age_acc = tm.functional.accuracy(age_logits, ages, task="multiclass")
            gender_acc = self.gender_acc(torch.sigmoid(gender_logits), genders)
            age_acc = self.age_acc(torch.sigmoid(age_logits), ages)
            adv_loss = gender_loss + age_loss
            outputs = {
                "sim": sim,
                "features": [f1, f2],
                "total_loss": contrastive_loss + adv_loss,
                "contrastive_loss": contrastive_loss,
                "adv_loss": adv_loss,
                "gender_loss": gender_loss,
                "age_loss": age_loss,
                "gender_acc": gender_acc,
                "age_acc": age_acc,
            }
            return outputs
        else:
            outputs = {"contrastive_loss": contrastive_loss, "sim": sim, "features": [f1, f2]}
            return outputs

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch, "train")
        total_loss = outputs["total_loss"]
        con_loss = outputs["contrastive_loss"]
        gender_loss = outputs["gender_loss"]
        age_loss = outputs["age_loss"]
        adv_loss = outputs["adv_loss"]
        gender_acc = outputs["gender_acc"]
        age_acc = outputs["gender_acc"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", con_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/total", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/gender", gender_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/age", age_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/adv", adv_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy/gender", gender_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy/age", age_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def _eval_step(self, batch, batch_idx, stage):
        _, _, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step(batch, stage)
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)
