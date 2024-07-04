import torch
import torch.nn.functional as F


def contrastive_loss(x1, x2, beta=0.08):
    x1x2 = torch.cat([x1, x2], dim=0)
    x2x1 = torch.cat([x2, x1], dim=0)
    cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1), torch.unsqueeze(x1x2, dim=0), dim=2) / beta
    mask = 1.0 - torch.eye(2 * x1.size(0)).to(x1.device)
    numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / beta)
    denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
    return -torch.mean(torch.log(numerators / denominators), dim=0)


class ContrastiveLossWithLabels(torch.nn.Module):

    def __init__(self, beta=0.08):
        super().__init__()
        self.beta = beta

    def forward(self, embeddings, positive_pairs):
        """
        Compute the contrastive loss term.

        Args:
            embeddings (torch.Tensor): The embeddings of the batch, shape (batch_size, embedding_dim)
            positive_pairs (list of tuples): List of tuples indicating positive pairs indices.

        Returns:
            torch.Tensor: The contrastive loss term.
        """
        batch_size = embeddings.size(0)
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        # Create masks to exclude self-similarities and positive pairs
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        for i, j in positive_pairs:
            mask[i, j] = True
            mask[j, i] = True

        exp_cosine_sim = torch.exp(cosine_sim / self.beta)
        exp_cosine_sim_masked = exp_cosine_sim.masked_fill(mask, 0)

        contrastive_loss = 0
        num_pairs = len(positive_pairs)

        for i, j in positive_pairs:
            pos_sim_ij = exp_cosine_sim[i, j]
            pos_sim_ji = exp_cosine_sim[j, i]

            neg_sims_i = exp_cosine_sim_masked[i]  # Exclude self and positive pairs
            neg_sims_j = exp_cosine_sim_masked[j]  # Exclude self and positive pairs

            loss_ij = -torch.log(pos_sim_ij / (pos_sim_ij + torch.sum(neg_sims_i)))
            loss_ji = -torch.log(pos_sim_ji / (pos_sim_ji + torch.sum(neg_sims_j)))

            contrastive_loss += loss_ij + loss_ji

        contrastive_loss /= 2 * num_pairs  # Average the loss over both directions

        return contrastive_loss


class HardContrastiveLoss(torch.nn.Module):

    def __init__(self, beta=0.08, alpha=0.8):
        super().__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, embeddings, positive_pairs):
        """
        Compute the contrastive loss term.

        Args:
            embeddings (torch.Tensor): The embeddings of the batch, shape (batch_size, embedding_dim)
            positive_pairs (list of tuples): List of tuples indicating positive pairs indices.

        Returns:
            torch.Tensor: The contrastive loss term.
        """
        batch_size = embeddings.size(0)
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        # Create masks to exclude self-similarities and positive pairs
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        for i, j in positive_pairs:
            mask[i, j] = True
            mask[j, i] = True

        exp_cosine_sim = torch.exp(cosine_sim / self.beta)
        exp_cosine_sim_masked = exp_cosine_sim.masked_fill(mask, 0)

        contrastive_loss = 0
        num_pairs = len(positive_pairs)

        for i, j in positive_pairs:
            pos_sim_ij = exp_cosine_sim[i, j]
            pos_sim_ji = exp_cosine_sim[j, i]

            # Select hard negatives based on the alpha quantile
            threshold_i = torch.quantile(exp_cosine_sim_masked[i], self.alpha)
            threshold_j = torch.quantile(exp_cosine_sim_masked[j], self.alpha)

            hard_neg_sims_i = exp_cosine_sim_masked[i][exp_cosine_sim_masked[i] >= threshold_i]
            hard_neg_sims_j = exp_cosine_sim_masked[j][exp_cosine_sim_masked[j] >= threshold_j]

            loss_ij = -torch.log(pos_sim_ij / (pos_sim_ij + torch.sum(hard_neg_sims_i)))
            loss_ji = -torch.log(pos_sim_ji / (pos_sim_ji + torch.sum(hard_neg_sims_j)))

            contrastive_loss += loss_ij + loss_ji

        contrastive_loss /= 2 * num_pairs  # Average the loss over both directions

        return contrastive_loss
