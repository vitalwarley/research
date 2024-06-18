import lightning as L
import numpy as np
import pandas as pd
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
    def __init__(self, model: str, attention: nn.Module):
        super(FaCoR, self).__init__()
        self.backbone = load_pretrained_model(model)
        self.attention = attention

    def forward(self, imgs):
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
    """
    Similar to FaCoR. Different with respect to the attention. It is designed for KFCAttention.
    """

    def forward(self, imgs):
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
        f1s, f2s, *attention_map = self.attention(f1_0, x1_feat, f2_0, x2_feat)

        return f1s, f2s, attention_map


class FaCoRV3(FaCoR):
    """
    Traditional kinship verification only.

    It is designed for KFCAttention. Also differs from FaCoR in the ausence of l2_norm for the features..

    Note that there is a mistake: with KFCAttention, f1s and f2s are the before-conv attention maps.
    """

    def forward(self, imgs):
        img1, img2 = imgs
        idx = [2, 1, 0]
        f1_0, x1_feat = self.backbone(img1[:, idx])  # (B, 512) and (B, 512, 7, 7)
        f2_0, x2_feat = self.backbone(img2[:, idx])  # ...

        # (B, 512), (B, 512), (B, 49, 49)
        f1s, f2s, *attention_map = self.attention(f1_0, x1_feat, f2_0, x2_feat)

        return f1s, f2s, attention_map


class FaCoRV4(FaCoR):
    """
    Traditional kinship verification only.

    Similar to FaCoR, but also returns the backbone embeddings.

    """

    def forward(self, imgs):
        img1, img2 = imgs
        idx = [2, 1, 0]
        f1_0, x1_feat = self.backbone(img1[:, idx])  # (B, 512) and (B, 512, 7, 7)
        f2_0, x2_feat = self.backbone(img2[:, idx])  # ...

        # Both are (B, 512)
        f1_0 = l2_norm(f1_0)
        f2_0 = l2_norm(f2_0)

        f1s, f2s, attention_map = self.attention(f1_0, x1_feat, f2_0, x2_feat)

        return f1_0, f2_0, f1s, f2s, attention_map


class FaCoRV5(FaCoR):
    """
    Designed for FaCoRNetBasic.
    """

    def forward(self, imgs):
        img1, img2 = imgs
        idx = [2, 1, 0]
        f1_0, x1_feat = self.backbone(img1[:, idx])  # (B, 512) and (B, 512, 7, 7)
        f2_0, x2_feat = self.backbone(img2[:, idx])  # ...

        # Both are (B, 512)
        f1_0 = l2_norm(f1_0)
        f2_0 = l2_norm(f2_0)

        f1s, f2s = self.attention(f1_0, x1_feat, f2_0, x2_feat)

        return f1s, f2s


class FaCoRV6(FaCoR):
    """
    Designed for FaCoRNetBasic. Added projection head.
    """

    def __init__(self, **kwargs):
        super(FaCoRV6, self).__init__(**kwargs)
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)

    def forward(self, imgs):
        img1, img2 = imgs
        idx = [2, 1, 0]
        f1_0, x1_feat = self.backbone(img1[:, idx])  # (B, 512) and (B, 512, 7, 7)
        f2_0, x2_feat = self.backbone(img2[:, idx])  # ...

        # Both are (B, 512)
        f1_0 = l2_norm(f1_0)
        f2_0 = l2_norm(f2_0)

        f1s, f2s = self.attention(f1_0, x1_feat, f2_0, x2_feat)

        f1s = self.fc2(self.relu(self.fc1(f1s)))
        f2s = self.fc2(self.relu(self.fc1(f2s)))

        return f1s, f2s


class FaCoRV7(FaCoR):
    """
    Designed for FaCoRNetBasic and ArcFace.
    """

    def forward(self, imgs):
        img1, img2 = imgs
        # idx = [2, 1, 0]
        idx = [0, 1, 2]
        f1_0 = self.backbone(img1[:, idx])  # (B, 512) and (B, 512, 7, 7)
        f2_0 = self.backbone(img2[:, idx])  # ...

        # Both are (B, 512)
        # f1_0 = l2_norm(f1_0)
        # f2_0 = l2_norm(f2_0)

        return f1_0, f2_0


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


class FaCoRNetTask3(L.LightningModule):
    """
    Task 3 is a bit different.

    It uses the backbone embeddings to compute the similarity between the probes and the gallery.
    """

    def __init__(self, model: nn.Module, list_dir: str):
        super().__init__()
        self.model = model
        self.list_dir = list_dir
        self.similarity_data = []
        self.rank_min = []
        self.rank_max = []

    def setup(self, stage):
        if stage == "predict":
            self._read_lists(self.list_dir)

    def _read_lists(self, list_dir):
        self.probe_fids = self._load_fids(f"{list_dir}/probe_list.csv")
        self.gallery_fids = self._load_fids(f"{list_dir}/gallery_list.csv")
        self.n_probes = len(self.probe_fids)
        self.n_gallery = len(self.gallery_fids)

    def _load_fids(self, path):
        fids = pd.read_csv(path).fid.values
        fids = [int(fid[1:]) for fid in fids]
        return torch.tensor(fids).cuda()

    def compute_rank_k_accuracy(self, rank_matrix, k=1):
        """
        Compute the accuracy at rank k.
        Each row of rank_matrix represents a probe and contains the indices of gallery images sorted by similarity.
        """
        # Map the sorted gallery indices to their family IDs
        sorted_gallery_fids = self.gallery_fids[rank_matrix]

        # Compare the family IDs of probes with the family IDs of top-k sorted gallery images
        matches = sorted_gallery_fids[:, :k] == self.probe_fids[:, None]

        # Check if any of the top k entries is a match for each probe
        correct_matches = matches.any(dim=1)

        # Calculate the accuracy as the mean of correct matches
        accuracy = correct_matches.float().mean().item()
        return accuracy

    def compute_map(self, rank_matrix):
        """
        Compute the Mean Average Precision (mAP) across all probes.
        Each row of rank_matrix represents a probe and contains the indices of gallery images sorted by similarity.
        """
        # Map the sorted gallery indices to their family IDs
        sorted_gallery_fids = self.gallery_fids[rank_matrix]

        # Calculate matches
        relevance = (sorted_gallery_fids == self.probe_fids[:, None]).float()

        # Calculate precision at each rank
        cumsum = torch.cumsum(relevance, dim=1)
        precision_at_k = cumsum / torch.arange(1, relevance.shape[1] + 1).cuda()

        # Only consider the ranks where relevance is 1
        average_precision = (precision_at_k * relevance).sum(dim=1) / relevance.sum(dim=1)

        # Handle division by zero for cases with no relevant documents
        average_precision[torch.isnan(average_precision)] = 0

        # Compute mean of average precisions
        map_score = average_precision.mean().item()
        return map_score

    def forward(self, inputs):
        return self.model.backbone(inputs[:, [2, 1, 0]])[0]  # why? the original code has this.

    def predict_step(self, batch, batch_idx):
        (probe_index, probe_images), (gallery_ids, gallery_images) = batch
        probe_embeddings = self(probe_images)
        gallery_embeddings = self(gallery_images)
        similarities = torch.cosine_similarity(gallery_embeddings.unsqueeze(1), probe_embeddings.unsqueeze(0), dim=2)
        max_similarities, _ = similarities.max(dim=1)
        mean_similarities = similarities.mean(dim=1)
        self.similarity_data.append((gallery_ids, max_similarities, mean_similarities))
        return gallery_ids

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        if outputs[-1] == self.n_gallery - 1:  # Last batch
            self._compute_ranks()

    def _compute_ranks(self):
        gallery_indexes, max_similarities, mean_similarities = zip(*self.similarity_data)

        gallery_indexes = torch.cat(gallery_indexes)
        max_similarities = torch.cat(max_similarities)
        mean_similarities = torch.cat(mean_similarities)

        # Stack the gallery_indexes and each similarity
        max_similarities = torch.stack([gallery_indexes, max_similarities], dim=1)
        mean_similarities = torch.stack([gallery_indexes, mean_similarities], dim=1)
        # Sort the similarities
        max_indices = torch.argsort(max_similarities[:, 1], descending=True)
        mean_indices = torch.argsort(mean_similarities[:, 1], descending=True)

        rank_max = gallery_indexes[max_indices]
        rank_min = gallery_indexes[mean_indices]

        # Store the rank of the max and mean similarities gallery indexes
        self.rank_max.append(rank_max)
        self.rank_min.append(rank_min)

        self.similarity_data = []

    def on_predict_epoch_end(self):
        rank_max = torch.cat(self.rank_max).view(self.n_probes, -1)
        rank_min = torch.cat(self.rank_min).view(self.n_probes, -1)
        # Compute the accuracy at rank 1 and mAP for both max and mean aggregations
        acc_1_max = self.compute_rank_k_accuracy(rank_max, k=1)
        acc_5_max = self.compute_rank_k_accuracy(rank_max, k=5)
        acc_1_min = self.compute_rank_k_accuracy(rank_min, k=1)
        acc_5_min = self.compute_rank_k_accuracy(rank_min, k=5)
        map_max = self.compute_map(rank_max)
        map_min = self.compute_map(rank_min)
        # Print logs as key: value in one line
        print(
            f"\nrank@1/max: {acc_1_max:.4f}, rank@5/max: {acc_5_max:.4f}, "
            + f"rank@1/min: {acc_1_min:.4f}, rank@5/min: {acc_5_min:.4f}, "
            + f"mAP/max: {map_max:.4f}, mAP/min: {map_min:.4f}"
        )


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


class FaCoRNetKFC(LightningBaseModel):
    """
    Designed for KFC with traditional contrastive loss (ContrasiveLossV2).
    """

    def _step(self, inputs):
        f1, f2, *_ = self(inputs)
        loss = self.criterion(f1, f2)
        sim = torch.cosine_similarity(f1, f2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2]}
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, _ = batch
        outputs = self._step([img1, img2])
        loss = outputs["contrastive_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step([img1, img2])
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FaCoRNetKFCV2(LightningBaseModel):
    """
    Designed for KFC with traditional contrastive loss (ContrasiveLossV2).

    It uses backbone embeddings at validation and test time to compute the similarity.
    """

    def _step(self, inputs):
        e1, e2, f1, f2, *_ = self(inputs)
        loss = self.criterion(f1, f2)
        sim = torch.cosine_similarity(e1, e2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2]}
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, _ = batch
        outputs = self._step([img1, img2])
        loss = outputs["contrastive_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step([img1, img2])
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FaCoRNetBasic(LightningBaseModel):
    """
    Designed for traditional contrastive loss (ContrasiveLossV2). No attention mechanism.
    """

    def _step(self, inputs):
        f1, f2 = self(inputs)
        loss = self.criterion(f1, f2)
        sim = torch.cosine_similarity(f1, f2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2]}
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, _ = batch
        outputs = self._step([img1, img2])
        loss = outputs["contrastive_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step([img1, img2])
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FaCoRNetBasicV2(LightningBaseModel):
    """
    Designed for traditional contrastive loss (ContrasiveLossV2). No attention mechanism.

    Differs from V1 in the use of cross entropy loss for validation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bce = nn.BCEWithLogitsLoss()

    def _step(self, inputs):
        img1, img2, labels = inputs
        f1, f2 = self((img1, img2))
        loss = self.criterion(f1, f2)
        # Cross-entropy loss based on similarities and best threshold
        sim = torch.cosine_similarity(f1, f2)
        logits = torch.logit((sim + 1) / 2)
        bce_loss = self.bce(logits, labels.float())
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2], "bce_loss": bce_loss}
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, labels = batch
        _, is_kin = labels
        outputs = self._step((img1, img2, is_kin))
        con_loss = outputs["contrastive_loss"]
        bce_loss = outputs["bce_loss"]
        if self.loss_factor:
            total_loss = self.loss_factor * con_loss + (1 - self.loss_factor) * bce_loss
        else:
            total_loss = con_loss + bce_loss
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", con_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/bce", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/total", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step((img1, img2, is_kin))
        con_loss = outputs["contrastive_loss"]
        bce_loss = outputs["bce_loss"]
        if self.loss_factor:
            total_loss = self.loss_factor * con_loss + (1 - self.loss_factor) * bce_loss
        else:
            total_loss = con_loss + bce_loss
        self.log(f"loss/{stage}", con_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"loss/{stage}/bce", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"loss/{stage}/total", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FaCoRNetBasicV3(LightningBaseModel):
    """
    Designed for traditional contrastive loss (ContrasiveLossV2). No attention mechanism.

    Implements contrastive loss at kinship type-level discrimination.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proj_head = nn.Linear(512, 128)

        class Attention(nn.Module):
            def __init__(self, embed_dim, num_heads):
                super(Attention, self).__init__()
                self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
                self.layer_norm = nn.LayerNorm(embed_dim)
                self.fc = nn.Linear(embed_dim, embed_dim)

            def forward(self, x):
                x = x.unsqueeze(0)  # Add sequence dimension: (seq_len=1, batch_size, embed_dim)
                attn_output, _ = self.multihead_attn(x, x, x)
                x = self.layer_norm(attn_output + x)
                x = self.fc(x)
                return x.squeeze(0)  # Remove sequence dimension: (batch_size, embed_dim)

        self.attention = Attention(embed_dim=256, num_heads=4)

    def _fusion(self, inputs):
        f1, f2 = inputs
        features = torch.cat([f1, f2], dim=1)
        return self.attention(features)

    def _step(self, inputs):
        pair1, pair2, _ = inputs
        p1_f1, p1_f2 = self(pair1)
        p1_z1, p1_z2 = self.proj_head(p1_f1), self.proj_head(p1_f2)
        p2_f1, p2_f2 = self(pair2)
        p2_z1, p2_z2 = self.proj_head(p2_f1), self.proj_head(p2_f2)
        p1 = self._fusion([p1_z1, p1_z2])
        p2 = self._fusion([p2_z1, p2_z2])
        loss = self.criterion(p1, p2)
        # Cross-entropy loss based on similarities and best threshold
        sim = torch.cosine_similarity(p1, p2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [p1, p2]}
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        con_loss = outputs["contrastive_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", con_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return con_loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        f1, f2 = self((img1, img2))
        loss = self.criterion(f1, f2)
        sim = torch.cosine_similarity(f1, f2)
        self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(sim)
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FaCoRNetBasicV4(LightningBaseModel):
    """
    Designed for traditional contrastive loss (ContrasiveLossV2). No attention mechanism. Added projection head.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)

    def _step(self, inputs):
        f1, f2 = self(inputs)
        f1_proj = self.fc2(self.relu(self.fc1(f1)))
        f2_proj = self.fc2(self.relu(self.fc1(f2)))
        # norm_f1 = torch.nn.functional.normalize(f1_proj, p=2, dim=1)
        # norm_f2 = torch.nn.functional.normalize(f2_proj, p=2, dim=1)
        loss = self.criterion(f1_proj, f2_proj)
        sim = torch.cosine_similarity(f1, f2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2]}
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, _ = batch
        outputs = self._step([img1, img2])
        loss = outputs["contrastive_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step([img1, img2])
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


if __name__ == "__main__":
    from models.attention import FaCoRAttention

    from datasets.facornet import FaCoRNetDMTask3

    model = FaCoRNetTask3(
        model=FaCoR(model="ir_101", attention=FaCoRAttention()), list_dir="../datasets/rfiw2021-track3/txt"
    )
    model.setup("predict")
    model.eval()
    dm = FaCoRNetDMTask3(root_dir="../datasets/rfiw2021-track3")
    dm.setup("predict")
    sr_dataloader = dm.predict_dataloader()

    batch = next(iter(sr_dataloader))
    model.predict_step(batch, 0)
