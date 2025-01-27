import pandas as pd
import torch

from models.base import LightningBaseModel


class SCL(LightningBaseModel):
    """
    Same as FaCoRNetBasicV6.
    """

    def __init__(self, *args, enable_hcl_on=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_hcl_on = enable_hcl_on

    def _step(self, batch):
        images, labels = batch
        is_kin = self._get_is_kin(labels)
        f1, f2 = self._forward_pass(images)
        loss = self._compute_loss(f1, f2, is_kin)
        sim = self._compute_similarity(f1, f2)

        return {
            "contrastive_loss": loss,
            "sim": sim,
            "features": [f1, f2],
            "difficulty_scores": 1 - sim.detach(),
        }

    def _get_is_kin(self, labels):
        return labels[-1] if isinstance(labels, (list, tuple)) else labels

    def _forward_pass(self, images):
        return self(images)

    def _should_apply_hcl(self):
        """Determine if HCL should be applied based on current training state."""
        is_training = self.trainer.state.stage == "train"
        is_sanity_check = self.trainer.state.stage == "sanity_check"
        past_enable_epoch = self.trainer.current_epoch >= self.enable_hcl_on
        return (is_training or is_sanity_check) and past_enable_epoch

    def _compute_loss(self, f1, f2, is_kin):
        features = torch.cat([f1, f2], dim=0)
        n_samples = f1.size(0)
        positive_pairs = torch.tensor(
            [(i, i + n_samples) for i in range(n_samples) if is_kin[i]]
        )
        return self.criterion(features, positive_pairs, self._should_apply_hcl())

    def _compute_similarity(self, f1, f2):
        return torch.cosine_similarity(f1, f2)

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        self._log(outputs)
        loss = outputs["contrastive_loss"]

        # Update difficulty scores in the sampler
        if self.trainer.datamodule.train_sampler is not None:
            difficulty_scores = outputs["difficulty_scores"]
            sampler = self.trainer.datamodule.train_sampler
            # Update difficulty scores for each sample in batch
            for item_idx, sim_difficulty in enumerate(difficulty_scores):
                difficulty = sim_difficulty.item()
                sampler.update_difficulty_scores(item_idx, difficulty)

        return loss

    def _log(self, outputs):
        loss = outputs["contrastive_loss"]
        self._log_training_metrics(loss)

    def _log_training_metrics(self, loss):
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log(
            "loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Log CV metrics if sampler is available
        if self.trainer.datamodule.train_sampler is not None:
            cv_stats = self.trainer.datamodule.train_sampler.get_sampling_stats()
            for metric_name, value in cv_stats.items():
                self.log(
                    f"cv/{metric_name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

    def _eval_step(self, batch, batch_idx, stage):
        _, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step(batch)
        self._log_eval_metrics(outputs, stage, is_kin, kin_relation)

    def _log_eval_metrics(self, outputs, stage, is_kin, kin_relation):
        self.log(
            f"loss/{stage}",
            outputs["contrastive_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class SCLTask2(SCL):
    """
    Same as FaCoRNetBasicV6.
    """

    def _step(self, batch):
        images, labels = batch
        is_kin = self._get_is_kin(labels)
        f1, f2, f3 = self._forward_pass(images)  # father, mother, child
        fc_loss = self._compute_loss(f1, f3, is_kin)
        mc_loss = self._compute_loss(f2, f3, is_kin)
        fc_sim = self._compute_similarity(f1, f3)
        mc_sim = self._compute_similarity(f2, f3)
        sim = (fc_sim + mc_sim) / 2
        return {
            "contrastive_loss": [fc_loss, mc_loss],
            "sim": sim,
            "features": [f1, f2, f3],
            "difficulty_scores": 1
            - sim.detach(),  # Higher similarity = lower difficulty
        }

    def _get_is_kin(self, labels):
        return labels[-1] if isinstance(labels, (list, tuple)) else labels

    def _forward_pass(self, images):
        return self(images)

    def _compute_loss(self, f1, f2, is_kin):
        features = torch.cat([f1, f2], dim=0)
        n_samples = f1.size(0)
        positive_pairs = torch.tensor(
            [(i, i + n_samples) for i in range(n_samples) if is_kin[i]]
        )
        return self.criterion(features, positive_pairs, self.trainer.state.stage)

    def _compute_similarity(self, f1, f2):
        return torch.cosine_similarity(f1, f2)

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        loss = outputs["contrastive_loss"]
        self._log_training_metrics(loss)

        # Update difficulty scores in the sampler
        if self.trainer.datamodule.train_sampler is not None:
            difficulty_scores = outputs["difficulty_scores"]
            sampler = self.trainer.datamodule.train_sampler
            # Update difficulty scores for each sample in batch
            for item_idx, sim_difficulty in enumerate(difficulty_scores):
                difficulty = sim_difficulty.item()
                sampler.update_difficulty_scores(item_idx, difficulty)

        return sum(loss)

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
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class SCLTask3(SCL):
    """
    Task 3 is a bit different.

    It uses the backbone embeddings to compute the similarity between the probes and the gallery.
    """

    def setup(self, stage):
        self.similarity_data = []
        self.rank_mean = []
        self.rank_max = []
        self.cached_probe_embeddings = {}
        self.cached_gallery_embeddings = {}

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
        average_precision = (precision_at_k * relevance).sum(dim=1) / relevance.sum(
            dim=1
        )

        # Handle division by zero for cases with no relevant documents
        average_precision[torch.isnan(average_precision)] = 0

        # Compute mean of average precisions
        map_score = average_precision.mean().item()
        return map_score

    def forward(self, inputs):
        return self.model.backbone(inputs[:, [2, 1, 0]])[0]

    def predict_step(self, batch, batch_idx):
        # Unpack the batch containing probe and gallery data
        (probe_index, probe_images, gallery_ids, gallery_images) = batch

        # Cache probe embeddings to avoid recomputing for same probe images
        # Each probe may have multiple images that we need embeddings for
        if probe_index not in self.cached_probe_embeddings:
            probe_embeddings = self(probe_images)  # Get embeddings for all probe images
            self.cached_probe_embeddings[probe_index] = probe_embeddings
        else:
            probe_embeddings = self.cached_probe_embeddings[probe_index]

        # Cache gallery embeddings using batch range as key
        # This avoids recomputing embeddings for same gallery batch
        batch_key_gallery = f"{gallery_ids[0].item()}-{gallery_ids[-1].item()}"
        if batch_key_gallery not in self.cached_gallery_embeddings:
            gallery_embeddings = self(gallery_images)
            self.cached_gallery_embeddings[batch_key_gallery] = gallery_embeddings
        else:
            gallery_embeddings = self.cached_gallery_embeddings[batch_key_gallery]

        # Compute cosine similarity between gallery and probe embeddings
        # Unsqueeze adds dimensions to allow broadcasting:
        # gallery: [G, 1, D] and probe: [1, P, D] -> similarities: [G, P]
        similarities = torch.cosine_similarity(
            gallery_embeddings.unsqueeze(1), probe_embeddings.unsqueeze(0), dim=2
        )

        # Get max and mean similarities across probe images for each gallery image
        # This handles cases where a probe subject has multiple images
        max_similarities, _ = similarities.max(dim=1)  # Best match among probe images
        mean_similarities = similarities.mean(
            dim=1
        )  # Average match across probe images

        # Store similarity scores for later ranking computation
        self.similarity_data.append((gallery_ids, max_similarities, mean_similarities))
        return gallery_ids

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        if outputs[-1] == self.n_gallery - 1:  # Last batch
            self._compute_ranks()

    def _compute_ranks(self):
        gallery_indexes, max_similarities, mean_similarities = zip(
            *self.similarity_data
        )

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
        rank_mean = gallery_indexes[mean_indices]

        # Store the rank of the max and mean similarities gallery indexes
        self.rank_max.append(rank_max)
        self.rank_mean.append(rank_mean)

        self.similarity_data = []

    def on_predict_epoch_end(self):
        rank_max = torch.cat(self.rank_max).view(self.n_probes, -1)
        rank_mean = torch.cat(self.rank_mean).view(self.n_probes, -1)
        # Compute the accuracy at rank 1 and mAP for both max and mean aggregations
        acc_1_max = self.compute_rank_k_accuracy(rank_max, k=1)
        acc_5_max = self.compute_rank_k_accuracy(rank_max, k=5)
        acc_1_mean = self.compute_rank_k_accuracy(rank_mean, k=1)
        acc_5_mean = self.compute_rank_k_accuracy(rank_mean, k=5)
        map_max = self.compute_map(rank_max)
        map_mean = self.compute_map(rank_mean)
        # Print logs as key: value in one line
        print(
            f"\nrank@1/max: {acc_1_max:.4f}, rank@5/max: {acc_5_max:.4f}, "
            + f"rank@1/mean: {acc_1_mean:.4f}, rank@5/mean: {acc_5_mean:.4f}, "
            + f"mAP/max: {map_max:.4f}, mAP/mean: {map_mean:.4f}"
        )


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
        positive_pairs = torch.tensor(
            [(i, i + n_samples) for i in range(n_samples) if is_kin[i]]
        )
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
        self.log(
            "loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def _eval_step(self, batch, batch_idx, stage):
        _, _, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step(batch)
        self.log(
            f"loss/{stage}",
            outputs["contrastive_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)

    def on_train_epoch_end(self):
        self.criterion.update_temperature(
            self.trainer.current_epoch, self.trainer.max_epochs
        )
        self.log(
            "temperature",
            self.criterion.tau,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


class SCLRFIW2021(SCL):
    def _forward_pass(self, img1, img2):
        return self([img1, img2])[2:]
