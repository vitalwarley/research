import torch
import torch.nn.functional as F


class HCL(torch.nn.Module):
    """
    Hard Contrastive Loss (HCL) implementation

    This loss function extends the standard contrastive loss by introducing hard negative/positive mining.
    It filters out easy negative samples and only considers hard negatives for more effective training.

    Key Features:
    - Uses cosine similarity between embeddings
    - Implements hard negative mining using quantile-based thresholds
    - Optionally supports hard positive mining
    - Can be configured to enable/disable filtering via should_filter flag

    Args:
        tau (float): Temperature parameter for scaling similarities (default: 0.2)
        alpha_neg (float): Quantile threshold for hard negative mining (default: 0.8)
        alpha_pos (float): Quantile threshold for hard positive mining (default: 0.0)

    Note:
        This implementation computes thresholds on exp(sim_matrix / tau) instead of sim_matrix.
        This is mathematically equivalent since exp(x) is monotonically increasing:
        s_{i,k} > s_i ⟺ exp(s_{i,k}/τ) > exp(s_i/τ) for any positive τ.
    """

    def __init__(
        self,
        tau=0.2,
        alpha_neg=0.8,
        alpha_pos=0.0,
    ):
        super().__init__()
        self.tau = tau
        self.alpha_neg = alpha_neg
        self.alpha_pos = alpha_pos
        self.eps = 1e-8

    def forward(self, embeddings, positive_pairs, should_filter=False):
        sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        return self.compute_loss(sim_matrix, positive_pairs, should_filter)

    def compute_loss(self, sim_matrix, positive_pairs, should_filter):
        device = sim_matrix.device
        num_pairs = len(positive_pairs)
        indices_i, indices_j = positive_pairs.T

        # Create mask to exclude self-pairs and mark positive pairs
        mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
        mask[indices_i, indices_j] = True
        mask[indices_j, indices_i] = True

        # Calculate exponential similarities with temperature scaling
        exp_sim = torch.exp(sim_matrix / self.tau)
        exp_sim_masked = exp_sim.masked_fill(
            mask, 0
        )  # Zero out self-pairs and positive pairs

        # Extract positive pair similarities
        pos_exp_sim_ij = exp_sim[indices_i, indices_j]  # Similarities for i->j pairs
        pos_exp_sim_ji = exp_sim[indices_j, indices_i]  # Similarities for j->i pairs

        # Get all negative similarities for each anchor
        neg_exp_sim_i = exp_sim_masked[indices_i]  # Negative similarities for i anchors
        neg_exp_sim_j = exp_sim_masked[indices_j]  # Negative similarities for j anchors

        if should_filter:
            # Apply hard negative mining using quantile thresholds
            threshold_neg_i = torch.quantile(
                neg_exp_sim_i, self.alpha_neg, dim=1, keepdim=True
            )
            threshold_neg_j = torch.quantile(
                neg_exp_sim_j, self.alpha_neg, dim=1, keepdim=True
            )

            # Keep only hard negatives (those above threshold)
            hard_neg_exp_sims_i = torch.where(
                neg_exp_sim_i >= threshold_neg_i,
                neg_exp_sim_i,
                torch.tensor(0.0, device=device),
            )
            hard_neg_exp_sims_j = torch.where(
                neg_exp_sim_j >= threshold_neg_j,
                neg_exp_sim_j,
                torch.tensor(0.0, device=device),
            )

            # Sum hard negative similarities for denominator
            sum_hard_neg_exp_sims_i = hard_neg_exp_sims_i.sum(dim=1)
            sum_hard_neg_exp_sims_j = hard_neg_exp_sims_j.sum(dim=1)

            # Apply hard positive mining if alpha_pos > 0
            threshold_pos_i = torch.quantile(pos_exp_sim_ij, 1 - self.alpha_pos)
            threshold_pos_j = torch.quantile(pos_exp_sim_ji, 1 - self.alpha_pos)

            # Keep only hard positives (those below threshold)
            pos_exp_sim_ij = torch.where(
                pos_exp_sim_ij <= threshold_pos_i,
                pos_exp_sim_ij,
                torch.tensor(0.0, device=device),
            )
            pos_exp_sim_ji = torch.where(
                pos_exp_sim_ji <= threshold_pos_j,
                pos_exp_sim_ji,
                torch.tensor(0.0, device=device),
            )

            # Calculate NCE loss with hard mining
            loss_ij = -torch.log(
                (pos_exp_sim_ij + self.eps)
                / (pos_exp_sim_ij + sum_hard_neg_exp_sims_i + self.eps)
            )
            loss_ji = -torch.log(
                (pos_exp_sim_ji + self.eps)
                / (pos_exp_sim_ji + sum_hard_neg_exp_sims_j + self.eps)
            )
        else:
            # When filtering is disabled
            sum_neg_exp_sims_i = neg_exp_sim_i.sum(dim=1)
            sum_neg_exp_sims_j = neg_exp_sim_j.sum(dim=1)

            loss_ij = -torch.log(pos_exp_sim_ij / (pos_exp_sim_ij + sum_neg_exp_sims_i))
            loss_ji = -torch.log(pos_exp_sim_ji / (pos_exp_sim_ji + sum_neg_exp_sims_j))

        # Average the bidirectional losses
        contrastive_loss = (loss_ij + loss_ji).sum()
        return contrastive_loss / (2 * num_pairs)


# Not used in the paper
class HCLFT(HCL):
    def __init__(
        self,
        tau=0.2,
        alpha_neg=0.8,
        alpha_pos=0.0,
        gamma_ex=2.0,
        gamma_in=1.6,
        dim_mixing=False,
        normalize=False,
        inter_pos=False,
        extra_neg=False,
    ):
        super().__init__(tau, alpha_neg, alpha_pos)
        self.gamma_ex = gamma_ex
        self.gamma_in = gamma_in
        self.dim_mixing = dim_mixing
        self.normalize = normalize
        self.inter_pos = inter_pos
        self.extra_neg = extra_neg

    def forward(self, embeddings, positive_pairs, stage):
        original_sim = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )

        if self.gamma_ex or self.gamma_in:
            if self.gamma_ex:
                transformation = "ex" if not self.inter_pos else "in"
                hard_pos_embeddings = self.transform_pairs(
                    embeddings, positive_pairs, self.gamma_ex, transformation
                )
                hard_pos_sim = F.cosine_similarity(
                    hard_pos_embeddings.unsqueeze(1),
                    hard_pos_embeddings.unsqueeze(0),
                    dim=2,
                )
                original_sim[positive_pairs[:, 0], positive_pairs[:, 1]] = hard_pos_sim[
                    positive_pairs[:, 0], positive_pairs[:, 1]
                ]
                original_sim[positive_pairs[:, 1], positive_pairs[:, 0]] = hard_pos_sim[
                    positive_pairs[:, 1], positive_pairs[:, 0]
                ]
            if self.gamma_in:
                batch_size = embeddings.size(0)
                transformation = "in" if not self.extra_neg else "ex"
                negative_pairs = self.generate_negative_pairs(
                    batch_size, positive_pairs
                )
                hard_neg_embeddings = self.transform_pairs(
                    embeddings, negative_pairs, self.gamma_in, transformation
                )
                hard_neg_sim = F.cosine_similarity(
                    hard_neg_embeddings.unsqueeze(1),
                    hard_neg_embeddings.unsqueeze(0),
                    dim=2,
                )
                original_sim[negative_pairs[:, 0], negative_pairs[:, 1]] = hard_neg_sim[
                    negative_pairs[:, 0], negative_pairs[:, 1]
                ]
                original_sim[negative_pairs[:, 1], negative_pairs[:, 0]] = hard_neg_sim[
                    negative_pairs[:, 1], negative_pairs[:, 0]
                ]

        return self.compute_loss(original_sim, positive_pairs, stage)

    @staticmethod
    def generate_negative_pairs(batch_size, positive_pairs):
        all_pairs = set(
            (i, j) for i in range(batch_size) for j in range(batch_size) if i != j
        )
        positive_pairs_set = set((i.item(), j.item()) for i, j in positive_pairs)
        negative_pairs = list(all_pairs - positive_pairs_set)
        return torch.tensor(negative_pairs)

    def transform_pairs(self, embeddings, pairs, gamma, transformation):
        processed_embeddings = embeddings.clone()
        lambda_ = self.generate_lambda(
            gamma, embeddings.size(1) if self.dim_mixing else 1, embeddings.device
        )
        if transformation == "ex":
            lambda_ += 1  # Only for positive pairs

        i, j = pairs.T
        new_embedding_i = lambda_ * embeddings[i] + (1 - lambda_) * embeddings[j]
        new_embedding_j = lambda_ * embeddings[j] + (1 - lambda_) * embeddings[i]

        if self.normalize:
            new_embedding_i = F.normalize(new_embedding_i, p=2, dim=1)
            new_embedding_j = F.normalize(new_embedding_j, p=2, dim=1)

        processed_embeddings[i] = new_embedding_i
        processed_embeddings[j] = new_embedding_j
        return processed_embeddings

    def generate_lambda(self, value, size, device):
        return torch.distributions.Beta(value, value).sample((size,)).to(device)
