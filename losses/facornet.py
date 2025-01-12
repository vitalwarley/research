import torch
import torch.nn.functional as F


class FaCoRNetCL(torch.nn.Module):
    """
    Uses contrastive loss with labels and FaCoRNet beta.

    It enables the use of the FaCoRNet contrastive loss, which modifies the beta
    to be a derived parameter from the attention map.
    """

    def __init__(self, s=500):
        super().__init__()
        self.s = s

    def m(self, beta):
        beta = (beta**2).sum([1, 2]) / self.s
        return torch.cat([beta, beta]).reshape(-1)

    def forward(self, embeddings, attention_map, positive_pairs):
        """
        Compute the contrastive loss term.

        Args:
            embeddings (torch.Tensor): The embeddings of the batch, shape (batch_size, embedding_dim)
            positive_pairs (list of tuples): List of tuples indicating positive pairs indices.

        Returns:
            torch.Tensor: The contrastive loss term.
        """
        beta = self.m(attention_map)
        return self._compute_loss(embeddings, positive_pairs, beta)

    def _compute_loss(self, embeddings, positive_pairs, beta):
        """
        Compute the contrastive loss term.
        """
        device = embeddings.device
        num_pairs = len(positive_pairs)
        indices_i, indices_j = positive_pairs.T

        # Create mask for self-similarities and positive pairs
        mask = torch.eye(embeddings.size(0), dtype=torch.bool, device=device)
        mask[indices_i, indices_j] = True
        mask[indices_j, indices_i] = True

        # Calculate similarities and mask them
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        exp_sim = torch.exp(cosine_sim / beta.unsqueeze(1))  # Use beta instead of tau
        exp_sim_masked = exp_sim.masked_fill(mask, 0)

        # Get positive pair similarities
        pos_sim_ij = exp_sim[indices_i, indices_j]
        pos_sim_ji = exp_sim[indices_j, indices_i]

        # Get negative similarities for each anchor
        exp_i = exp_sim_masked[indices_i]
        exp_j = exp_sim_masked[indices_j]

        # Calculate loss using all negatives (no filtering needed as beta handles the scaling)
        sum_neg_sims_i = exp_i.sum(dim=1)
        sum_neg_sims_j = exp_j.sum(dim=1)

        loss_ij = -torch.log(pos_sim_ij / (pos_sim_ij + sum_neg_sims_i))
        loss_ji = -torch.log(pos_sim_ji / (pos_sim_ji + sum_neg_sims_j))

        contrastive_loss = (loss_ij + loss_ji).sum()
        return contrastive_loss / (2 * num_pairs)
