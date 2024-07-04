import torch
from models.base import LightningBaseModel


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
