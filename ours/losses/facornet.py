import torch
import torch.nn.functional as F
from scl import contrastive_loss


def facornet_contrastive_loss(x1, x2, beta=0.08):
    m = 0.0
    x1x2 = torch.cat([x1, x2], dim=0)
    x2x1 = torch.cat([x2, x1], dim=0)
    beta = (beta**2).sum([1, 2]) / 500
    beta = torch.cat([beta, beta]).reshape(-1)
    cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1), torch.unsqueeze(x1x2, dim=0), dim=2) / (beta + m)
    mask = 1.0 - torch.eye(2 * x1.size(0)).to(x1.device)
    numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / (beta + m))
    denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
    return -torch.mean(torch.log(numerators / denominators), dim=0)


class ContrastiveLossV2(torch.nn.Module):
    def __init__(self, s=0.08):
        super().__init__()
        self.beta = s

    def forward(self, x1, x2):
        return contrastive_loss(x1, x2, self.beta)


class FaCoRContrastiveLoss(torch.nn.Module):

    def __init__(self, s=500):
        super().__init__()
        self.s = s

    def m(self, beta):
        return (beta**2).sum([1, 2]) / self.s

    def forward(self, x1, x2, beta):
        m = 0.0
        x1x2 = torch.cat([x1, x2], dim=0)
        x2x1 = torch.cat([x2, x1], dim=0)
        beta = self.m(beta)
        beta = torch.cat([beta, beta]).reshape(-1)
        cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1), torch.unsqueeze(x1x2, dim=0), dim=2) / (
            beta + m
        )
        mask = 1.0 - torch.eye(2 * x1.size(0)).to(x1.device)
        numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / (beta + m))
        denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
        return -torch.mean(torch.log(numerators / denominators), dim=0)


class FaCoRContrastiveLossV2(FaCoRContrastiveLoss):

    def m(self, beta, epsilon=1e-6):
        beta = -(beta * (beta + epsilon).log()).sum(dim=[1, 2])
        return beta


class FaCoRContrastiveLossV3(FaCoRContrastiveLoss):

    def m(self, beta, epsilon=1e-6):
        beta = -(beta * (beta + epsilon).log()).sum(dim=[1, 2])
        n_features = beta.shape[-1]  # Assuming the last dim of attention holds the class probabilities
        max_entropy = torch.log(torch.tensor(n_features, dtype=torch.float32))
        normalized_entropy = beta / max_entropy

        return normalized_entropy


class FaCoRContrastiveLossV4(FaCoRContrastiveLoss):

    def m(self, beta):
        beta = beta.mean(dim=[1, 2])
        return beta


class FaCoRContrastiveLossV5(FaCoRContrastiveLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_val = 0.08  # Define your range minimum
        self.max_val = 0.1  # Define your range maximum

    def m(self, beta):
        # Normalize the tensor to [0, 1] and then scale to [min_val, max_val]
        beta = (beta**2).sum(dim=(1, 2))
        beta = (beta - beta.min()) / (beta.max() - beta.min())
        beta = beta * (self.max_val - self.min_val) + self.min_val
        return beta


class FaCoRContrastiveLossV6(FaCoRContrastiveLoss):

    def m(self, betas):
        beta_0, beta_1 = betas
        beta_0 = beta_0.mean(dim=[1, 2])
        beta_1 = beta_1.mean(dim=[1, 2])
        beta = beta_0 + beta_1  # / 2
        return beta


class FaCoRContrastiveLossV7(FaCoRContrastiveLoss):

    def m(self, betas):
        beta_0, beta_1 = betas
        beta_0 = (beta_0**2).sum([1, 2]) / self.s
        beta_1 = (beta_1**2).sum([1, 2]) / self.s
        beta = beta_0 + beta_1  # / 2
        return beta


class FaCoRNetCL(torch.nn.Module):

    def __init__(self, s=500):
        super().__init__()
        self.s = s

    def _contrastive_loss(self, x1, x2, beta):
        m = 0.0
        x1x2 = torch.cat([x1, x2], dim=0)
        x2x1 = torch.cat([x2, x1], dim=0)
        beta = self.m(beta)
        beta = torch.cat([beta, beta]).reshape(-1)
        cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1), torch.unsqueeze(x1x2, dim=0), dim=2) / (
            beta + m
        )
        mask = 1.0 - torch.eye(2 * x1.size(0)).to(x1.device)
        numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / (beta + m))
        denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
        return -torch.mean(torch.log(numerators / denominators), dim=0)

    def m(self, beta):
        return (beta**2).sum([1, 2]) / self.s

    def forward(self, x1, x2, attention_map):
        """
        Compute the contrastive loss term.

        Args:
            embeddings (torch.Tensor): The embeddings of the batch, shape (batch_size, embedding_dim)
            positive_pairs (list of tuples): List of tuples indicating positive pairs indices.

        Returns:
            torch.Tensor: The contrastive loss term.
        """
        return self._contrastive_loss(x1, x2, attention_map)
