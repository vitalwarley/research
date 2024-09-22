import torch
from models.base import LightningBaseModel


class SCL(LightningBaseModel):
    """
    Same as FaCoRNetBasicV6.
    """

    def _step(self, batch):
        img1, img2, labels = batch
        is_kin = self._get_is_kin(labels)
        f1, f2 = self._forward_pass(img1, img2)
        loss = self._compute_loss(f1, f2, is_kin)
        sim = self._compute_similarity(f1, f2)
        return {"contrastive_loss": loss, "sim": sim, "features": [f1, f2]}

    def _get_is_kin(self, labels):
        return labels[-1] if isinstance(labels, (list, tuple)) else labels

    def _forward_pass(self, img1, img2):
        return self([img1, img2])

    def _compute_loss(self, f1, f2, is_kin):
        features = torch.cat([f1, f2], dim=0)
        n_samples = f1.size(0)
        positive_pairs = torch.tensor([(i, i + n_samples) for i in range(n_samples) if is_kin[i]])
        return self.criterion(features, positive_pairs, self.trainer.state.stage)

    def _compute_similarity(self, f1, f2):
        return torch.cosine_similarity(f1, f2)

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        loss = outputs["contrastive_loss"]
        self._log_training_metrics(loss)
        return loss

    def _log_training_metrics(self, loss):
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _eval_step(self, batch, batch_idx, stage):
        _, _, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step(batch)
        self._log_eval_metrics(outputs, stage, is_kin, kin_relation)

    def _log_eval_metrics(self, outputs, stage, is_kin, kin_relation):
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
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


class SCLRFIW2021(SCL):

    def _forward_pass(self, img1, img2):
        return self([img1, img2])[2:]
