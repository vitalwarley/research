import torch
import torch.nn as nn

from models.base import load_pretrained_model
from models.scl import SCL, SCLTask3


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
        positive_pairs = torch.tensor(
            [(i, i + n_samples) for i in range(n_samples) if is_kin[i]]
        )
        return self.criterion(features, att, positive_pairs)

    def _step(self, batch):
        images, labels = batch
        is_kin = self._get_is_kin(labels)
        f1, f2, att = self._forward_pass(images)
        loss = self._compute_loss(f1, f2, is_kin, att)
        sim = self._compute_similarity(f1, f2)
        psi = self.criterion.m(att)
        return {
            "contrastive_loss": loss,
            "sim": sim,
            "features": [f1, f2, att],
            "psi": psi,
        }

    def _log(self, outputs):
        super()._log(outputs)
        # Add psi logging
        psi = outputs["psi"]
        self.logger.experiment.add_histogram("psi/train", psi, self.global_step)

    def _log_eval_metrics(self, outputs, stage, is_kin, kin_relation):
        super()._log_eval_metrics(outputs, stage, is_kin, kin_relation)
        # Add psi logging
        self.logger.experiment.add_histogram(
            f"psi/{stage}", outputs["psi"], self.global_step
        )

    def _eval_step(self, batch, batch_idx, stage):
        _, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step(batch)
        self._log_eval_metrics(outputs, stage, is_kin, kin_relation)
        psi = outputs["psi"]
        self.logger.experiment.add_histogram("psi/val", psi, self.global_step)


class FaCoRNetTask2(FaCoRNet):
    """
    Task 2 variant of FaCoRNet for parent-child verification.
    Handles father-child and mother-child pairs separately.
    """

    def _forward_pass(self, images):
        father, mother, child = images
        # Process father-child pair
        f1, f3_fc, att_fc = self([father, child])
        # Process mother-child pair
        f2, f3_mc, att_mc = self([mother, child])
        return f1, f2, (f3_fc, f3_mc), att_fc, att_mc

    def _step(self, batch):
        images, labels = batch
        is_kin = self._get_is_kin(labels)
        f1, f2, (f3_fc, f3_mc), att_fc, att_mc = self._forward_pass(
            images
        )  # father, mother, child
        fc_loss = self._compute_loss(f1, f3_fc, is_kin, att_fc)
        mc_loss = self._compute_loss(f2, f3_mc, is_kin, att_mc)
        fc_sim = self._compute_similarity(f1, f3_fc)
        mc_sim = self._compute_similarity(f2, f3_mc)
        sim = (fc_sim + mc_sim) / 2
        psi_fc = self.criterion.m(att_fc)
        psi_mc = self.criterion.m(att_mc)
        return {
            "contrastive_loss": [fc_loss, mc_loss],
            "sim": sim,
            "features": [f1, f2, (f3_fc, f3_mc), att_fc, att_mc],
            "psi": [psi_fc, psi_mc],
            "difficulty_scores": 1 - sim.detach(),
        }

    def _log_training_metrics(self, loss):
        fc_loss, mc_loss = loss
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log(
            "loss/train/fc",
            fc_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss/train/mc",
            mc_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        loss = outputs["contrastive_loss"]
        self._log_training_metrics(loss)
        psi_fc, psi_mc = outputs["psi"]
        self.logger.experiment.add_histogram("psi/train/fc", psi_fc, self.global_step)
        self.logger.experiment.add_histogram("psi/train/mc", psi_mc, self.global_step)

        return sum(loss)

    def _eval_step(self, batch, batch_idx, stage):
        _, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step(batch)
        self._log_eval_metrics(outputs, stage, is_kin, kin_relation)

    def _log_eval_metrics(self, outputs, stage, is_kin, kin_relation):
        self.log(
            f"loss/{stage}/fc",
            outputs["contrastive_loss"][0],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"loss/{stage}/mc",
            outputs["contrastive_loss"][1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        psi_fc, psi_mc = outputs["psi"]
        self.logger.experiment.add_histogram(
            f"psi/{stage}/fc", psi_fc, self.global_step
        )
        self.logger.experiment.add_histogram(
            f"psi/{stage}/mc", psi_mc, self.global_step
        )
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FaCoRNetTask3(SCLTask3):
    """
    Task 3 variant of FaCoRNet for family search.
    Uses backbone embeddings to compute similarity between probes and gallery.
    """

    def setup(self, stage):
        super().setup(stage)
        self.cached_probe_embeddings_feat = {}
        self.cached_gallery_embeddings_feat = {}

    def forward(self, inputs):
        f1_0, x1_feat = self.model.backbone(
            inputs[:, [2, 1, 0]]
        )  # (B, 512) and (B, 512, 7, 7)
        f1_0 = self.model.l2norm(f1_0)
        return f1_0, x1_feat

    # It may be possible to use attention yet, but it is not straightforwad.
    # attention needs x1 and x2 of the same length, but the probe images are of varying length.
    # gallery_images depends on the batch.
    # The FaCoRNet authors didn't specify how they did it. I won't use it.
    def predict_step(self, batch, batch_idx):
        (probe_index, probe_images, gallery_ids, gallery_images) = batch
        if probe_index not in self.cached_probe_embeddings:
            probe_embeddings_f1, probe_embeddings_feat = self(probe_images)
            self.cached_probe_embeddings[probe_index] = probe_embeddings_f1
            self.cached_probe_embeddings_feat[probe_index] = probe_embeddings_feat
        else:
            probe_embeddings_f1 = self.cached_probe_embeddings[probe_index]
            probe_embeddings_feat = self.cached_probe_embeddings_feat[probe_index]

        batch_key_gallery = f"{gallery_ids[0].item()}-{gallery_ids[-1].item()}"
        if batch_key_gallery not in self.cached_gallery_embeddings:
            gallery_embeddings_f1, gallery_embeddings_feat = self(gallery_images)
            self.cached_gallery_embeddings[batch_key_gallery] = gallery_embeddings_f1
            self.cached_gallery_embeddings_feat[batch_key_gallery] = (
                gallery_embeddings_feat
            )
        else:
            gallery_embeddings_f1 = self.cached_gallery_embeddings[batch_key_gallery]
            gallery_embeddings_feat = self.cached_gallery_embeddings_feat[
                batch_key_gallery
            ]

        similarities = torch.cosine_similarity(
            gallery_embeddings_f1.unsqueeze(1), probe_embeddings_f1.unsqueeze(0), dim=2
        )

        # For each probe, get max and mean similarities
        max_similarities, _ = similarities.max(dim=1)
        mean_similarities = similarities.mean(dim=1)

        self.similarity_data.append((gallery_ids, max_similarities, mean_similarities))
        return gallery_ids
