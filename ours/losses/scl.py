import torch
import torch.nn.functional as F


class MCLoss(torch.nn.Module):

    def __init__(self, alpha=0.2, beta=3.1, lambd=1.5, epsilon=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
        self.epsilon = epsilon

    def forward(self, x, y):
        # Compute M
        M = torch.matmul(x, y.t())

        # Filter positive samples
        M_pos = M.diag()
        M_neg = M[~torch.eye(M.size(0), dtype=torch.bool)].view(M.size(0), -1)

        valid_pos_mask = M_pos - self.epsilon < M_neg.max(dim=1)[0]
        valid_neg_mask = M_neg + self.epsilon > M_pos.min()

        # Compute positive loss
        pos_loss = (
            (1 / x.size(0))
            * (1 / self.alpha)
            * torch.log(1 + torch.sum(torch.exp(-self.alpha * (M_pos[valid_pos_mask] - self.lambd))))
        )

        # Compute negative loss
        neg_loss = (
            (1 / x.size(0))
            * (1 / self.beta)
            * torch.log(1 + torch.sum(torch.exp(self.beta * (M_neg[valid_neg_mask] - self.lambd))))
        )

        # Combine losses
        total_loss = pos_loss + neg_loss
        return total_loss


def contrastive_loss(x1, x2, beta=0.08):
    x1x2 = torch.cat([x1, x2], dim=0)
    x2x1 = torch.cat([x2, x1], dim=0)
    cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1), torch.unsqueeze(x1x2, dim=0), dim=2) / beta
    mask = 1.0 - torch.eye(2 * x1.size(0)).to(x1.device)
    numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / beta)
    denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
    return -torch.mean(torch.log(numerators / denominators), dim=0)


def contrastive_loss_with_labels(embeddings, positive_pairs, beta=0.08):
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

    exp_cosine_sim = torch.exp(cosine_sim / beta)
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
        return contrastive_loss_with_labels(embeddings, positive_pairs, self.beta)


class HardContrastiveLoss(torch.nn.Module):

    def __init__(self, beta=0.2, alpha=0.8):
        super().__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, embeddings, positive_pairs, stage):
        """
        Compute the contrastive loss term.

        Args:
            embeddings (torch.Tensor): The embeddings of the batch, shape (batch_size, embedding_dim)
            positive_pairs (list of tuples): List of tuples indicating positive pairs indices.

        Returns:
            torch.Tensor: The contrastive loss term.
        """
        batch_size = embeddings.size(0)
        num_pairs = len(positive_pairs)

        # Create masks to exclude self-similarities and positive pairs
        mask = torch.eye(batch_size, device=embeddings.device).bool()

        indices_i = positive_pairs[:, 0]
        indices_j = positive_pairs[:, 1]

        # Set the mask for positive pairs
        mask[indices_i, indices_j] = True
        mask[indices_j, indices_i] = True

        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        exp_cosine_sim = torch.exp(cosine_sim / self.beta)
        exp_cosine_sim_masked = exp_cosine_sim.masked_fill(mask, 0)

        pos_sim_ij = exp_cosine_sim[indices_i, indices_j]
        pos_sim_ji = exp_cosine_sim[indices_j, indices_i]

        exp_i = exp_cosine_sim_masked[indices_i]
        exp_j = exp_cosine_sim_masked[indices_j]

        if stage in ["train", "sanity_check"]:
            # Select hard negatives based on the alpha quantile
            threshold_i = torch.quantile(exp_i, self.alpha, dim=1, keepdim=True)
            threshold_j = torch.quantile(exp_j, self.alpha, dim=1, keepdim=True)

            hard_neg_sims_i = torch.where(exp_i >= threshold_i, exp_i, torch.tensor(0.0, device=embeddings.device))
            hard_neg_sims_j = torch.where(exp_j >= threshold_j, exp_j, torch.tensor(0.0, device=embeddings.device))

            sum_hard_neg_sims_i = hard_neg_sims_i.sum(dim=1)
            sum_hard_neg_sims_j = hard_neg_sims_j.sum(dim=1)

            loss_ij = -torch.log(pos_sim_ij / (pos_sim_ij + sum_hard_neg_sims_i))
            loss_ji = -torch.log(pos_sim_ji / (pos_sim_ji + sum_hard_neg_sims_j))
        else:
            sum_neg_sims_i = exp_i.sum(dim=1)
            sum_neg_sims_j = exp_j.sum(dim=1)

            loss_ij = -torch.log(pos_sim_ij / (pos_sim_ij + sum_neg_sims_i))
            loss_ji = -torch.log(pos_sim_ji / (pos_sim_ji + sum_neg_sims_j))

        contrastive_loss = (loss_ij + loss_ji).sum()
        contrastive_loss /= 2 * num_pairs  # Average the loss over both directions

        return contrastive_loss


class HardContrastiveLossV2(torch.nn.Module):

    def __init__(self, beta=0.2, alpha_neg=0.8, alpha_pos=0.2):
        super().__init__()
        self.beta = beta
        self.alpha_neg = alpha_neg
        self.alpha_pos = alpha_pos

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
        num_pairs = len(positive_pairs)

        # Create masks to exclude self-similarities and positive pairs
        mask = torch.eye(batch_size, device=embeddings.device).bool()

        indices_i = positive_pairs[:, 0]
        indices_j = positive_pairs[:, 1]

        # Set the mask for positive pairs
        mask[indices_i, indices_j] = True
        mask[indices_j, indices_i] = True

        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        exp_cosine_sim = torch.exp(cosine_sim / self.beta)
        exp_cosine_sim_masked = exp_cosine_sim.masked_fill(mask, 0)

        pos_sim_ij = exp_cosine_sim[indices_i, indices_j]
        pos_sim_ji = exp_cosine_sim[indices_j, indices_i]

        exp_i = exp_cosine_sim_masked[indices_i]
        exp_j = exp_cosine_sim_masked[indices_j]

        # Select hard negatives based on the alpha quantile
        threshold_i_neg = torch.quantile(exp_i, self.alpha_neg, dim=1, keepdim=True)
        threshold_j_neg = torch.quantile(exp_j, self.alpha_neg, dim=1, keepdim=True)

        hard_neg_sims_i = torch.where(exp_i >= threshold_i_neg, exp_i, torch.tensor(0.0, device=embeddings.device))
        hard_neg_sims_j = torch.where(exp_j >= threshold_j_neg, exp_j, torch.tensor(0.0, device=embeddings.device))

        sum_hard_neg_sims_i = hard_neg_sims_i.sum(dim=1)
        sum_hard_neg_sims_j = hard_neg_sims_j.sum(dim=1)

        # Select easy positives based on the alpha_positive quantile
        threshold_i_pos = torch.quantile(pos_sim_ij, self.alpha_pos, dim=0, keepdim=True)
        threshold_j_pos = torch.quantile(pos_sim_ji, self.alpha_pos, dim=0, keepdim=True)

        hard_pos_sim_ij = pos_sim_ij[pos_sim_ij <= threshold_i_pos]
        hard_pos_sim_ji = pos_sim_ji[pos_sim_ji <= threshold_j_pos]

        # Compute loss
        contrastive_loss = 0

        # TODO: is it right?
        for pos_sim_ij, pos_sim_ji in zip(hard_pos_sim_ij, hard_pos_sim_ji):

            loss_ij = -torch.log(pos_sim_ij / (pos_sim_ij + sum_hard_neg_sims_i))
            loss_ji = -torch.log(pos_sim_ji / (pos_sim_ji + sum_hard_neg_sims_j))

            contrastive_loss += (loss_ij + loss_ji).sum()

        contrastive_loss /= 2 * num_pairs  # Average the loss over both directions

        return contrastive_loss


class HardContrastiveLossV3(torch.nn.Module):

    def __init__(self, initial_tau=0.2, final_tau=0.02, alpha=0.8):
        super().__init__()
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.alpha = alpha
        self.tau = initial_tau

    def update_temperature(self, epoch, max_epochs):
        epoch = torch.tensor(epoch, dtype=torch.uint8)
        max_epochs = torch.tensor(max_epochs, dtype=torch.uint8)
        self.tau = self.initial_tau + (torch.log(epoch + 1) / torch.log(max_epochs + 1)) * (
            self.final_tau - self.initial_tau
        )

    def forward(self, embeddings, positive_pairs, stage):
        """
        Compute the contrastive loss term.

        Args:
            embeddings (torch.Tensor): The embeddings of the batch, shape (batch_size, embedding_dim)
            positive_pairs (list of tuples): List of tuples indicating positive pairs indices.

        Returns:
            torch.Tensor: The contrastive loss term.
        """
        batch_size = embeddings.size(0)
        num_pairs = len(positive_pairs)

        # Create masks to exclude self-similarities and positive pairs
        mask = torch.eye(batch_size, device=embeddings.device).bool()

        indices_i = positive_pairs[:, 0]
        indices_j = positive_pairs[:, 1]

        # Set the mask for positive pairs
        mask[indices_i, indices_j] = True
        mask[indices_j, indices_i] = True

        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        exp_cosine_sim = torch.exp(cosine_sim / self.tau)
        exp_cosine_sim_masked = exp_cosine_sim.masked_fill(mask, 0)

        pos_sim_ij = exp_cosine_sim[indices_i, indices_j]
        pos_sim_ji = exp_cosine_sim[indices_j, indices_i]

        exp_i = exp_cosine_sim_masked[indices_i]
        exp_j = exp_cosine_sim_masked[indices_j]

        if stage in ["train", "sanity_check"]:
            # Select hard negatives based on the alpha quantile
            threshold_i = torch.quantile(exp_i, self.alpha, dim=1, keepdim=True)
            threshold_j = torch.quantile(exp_j, self.alpha, dim=1, keepdim=True)

            hard_neg_sims_i = torch.where(exp_i >= threshold_i, exp_i, torch.tensor(0.0, device=embeddings.device))
            hard_neg_sims_j = torch.where(exp_j >= threshold_j, exp_j, torch.tensor(0.0, device=embeddings.device))

            sum_hard_neg_sims_i = hard_neg_sims_i.sum(dim=1)
            sum_hard_neg_sims_j = hard_neg_sims_j.sum(dim=1)

            loss_ij = -torch.log(pos_sim_ij / (pos_sim_ij + sum_hard_neg_sims_i))
            loss_ji = -torch.log(pos_sim_ji / (pos_sim_ji + sum_hard_neg_sims_j))
        else:
            sum_neg_sims_i = exp_i.sum(dim=1)
            sum_neg_sims_j = exp_j.sum(dim=1)

            loss_ij = -torch.log(pos_sim_ij / (pos_sim_ij + sum_neg_sims_i))
            loss_ji = -torch.log(pos_sim_ji / (pos_sim_ji + sum_neg_sims_j))

        contrastive_loss = (loss_ij + loss_ji).sum()
        contrastive_loss /= 2 * num_pairs  # Average the loss over both directions

        return contrastive_loss
