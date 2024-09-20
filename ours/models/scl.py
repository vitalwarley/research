import torch
from models.base import LightningBaseModel


class SCL(LightningBaseModel):
    """
    Same as FaCoRNetBasicV6.
    """

    def _step(self, batch):
        img1, img2, labels = batch
        if isinstance(labels, list | tuple):
            is_kin = labels[-1]
        else:
            is_kin = labels
        f1, f2 = self([img1, img2])
        features = torch.cat([f1, f2], dim=0)
        n_samples = f1.size(0)
        positive_pairs = torch.tensor([(i, i + n_samples) for i in range(n_samples) if is_kin[i]])
        loss = self.criterion(features, positive_pairs, self.trainer.state.stage)
        sim = torch.cosine_similarity(f1, f2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2]}
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        loss = outputs["contrastive_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def _eval_step(self, batch, batch_idx, stage):
        _, _, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step(batch)
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class SCLV2(LightningBaseModel):
    """
    Same as FaCoRNetBasicV6. With adjustable temperature.
    """

    def _step(self, batch):
        img1, img2, labels = batch
        if isinstance(labels, list | tuple):
            is_kin = labels[-1]
        else:
            is_kin = labels
        f1, f2 = self([img1, img2])
        features = torch.cat([f1, f2], dim=0)
        n_samples = f1.size(0)
        positive_pairs = torch.tensor([(i, i + n_samples) for i in range(n_samples) if is_kin[i]])
        loss = self.criterion(features, positive_pairs, self.trainer.state.stage)
        sim = torch.cosine_similarity(f1, f2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2]}
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        loss = outputs["contrastive_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _eval_step(self, batch, batch_idx, stage):
        _, _, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step(batch)
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)

    def on_train_epoch_end(self):
        self.criterion.update_temperature(self.trainer.current_epoch, self.trainer.max_epochs)
        self.log("temperature", self.criterion.tau, on_step=False, on_epoch=True, prog_bar=True, logger=True)
