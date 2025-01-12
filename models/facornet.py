import torch
import torch.nn as nn

from models.base import load_pretrained_model
from models.scl import SCL


# Define a custom L2 normalization layer
class L2Norm(nn.Module):
    def __init__(self, axis=1):
        super(L2Norm, self).__init__()
        self.axis = axis

    def forward(self, x):
        # L2 normalization
        return nn.functional.normalize(x, p=2, dim=self.axis)


class FaCoR(torch.nn.Module):
    def __init__(self, model: str, attention: nn.Module):
        super(FaCoR, self).__init__()
        self.backbone = load_pretrained_model(model)
        self.attention = attention
        self.l2norm = L2Norm()

    def forward(self, imgs):
        img1, img2 = imgs
        idx = [2, 1, 0]
        f1_0, x1_feat = self.backbone(img1[:, idx])  # (B, 512) and (B, 512, 7, 7)
        f2_0, x2_feat = self.backbone(img2[:, idx])  # ...

        # Both are (B, 512)
        f1_0 = self.l2norm(f1_0)
        f2_0 = self.l2norm(f2_0)

        f1s, f2s, attention_map = self.attention(f1_0, x1_feat, f2_0, x2_feat)

        return f1s, f2s, attention_map


class FaCoRNet(SCL):
    """
    Similar to SCL but uses FaCoRNetCL criterion which expects attention maps.
    """

    def _compute_loss(self, f1, f2, is_kin, att):
        features = torch.cat([f1, f2], dim=0)
        n_samples = f1.size(0)
        positive_pairs = torch.tensor([(i, i + n_samples) for i in range(n_samples) if is_kin[i]])
        return self.criterion(features, att, positive_pairs)

    def _step(self, batch):
        images, labels = batch
        is_kin = self._get_is_kin(labels)
        f1, f2, att = self._forward_pass(images)
        loss = self._compute_loss(f1, f2, is_kin, att)
        sim = self._compute_similarity(f1, f2)
        psi = self.criterion.m(att)
        return {"contrastive_loss": loss, "sim": sim, "features": [f1, f2, att], "psi": psi}

    def _log(self, outputs):
        super()._log(outputs)
        # Add psi logging
        psi = outputs["psi"]
        self.logger.experiment.add_histogram("psi/train", psi, self.global_step)

    def _log_eval_metrics(self, outputs, stage, is_kin, kin_relation):
        super()._log_eval_metrics(outputs, stage, is_kin, kin_relation)
        # Add psi logging
        self.logger.experiment.add_histogram(f"psi/{stage}", outputs["psi"], self.global_step)

    def _eval_step(self, batch, batch_idx, stage):
        _, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step(batch)
        self._log_eval_metrics(outputs, stage, is_kin, kin_relation)
        psi = outputs["psi"]
        self.logger.experiment.add_histogram("psi/val", psi, self.global_step)
