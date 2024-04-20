import torch
import torch.nn.functional as F


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


def contrastive_loss(x1, x2, beta=0.08):
    x1x2 = torch.cat([x1, x2], dim=0)
    x2x1 = torch.cat([x2, x1], dim=0)
    cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1), torch.unsqueeze(x1x2, dim=0), dim=2) / beta
    mask = 1.0 - torch.eye(2 * x1.size(0)).to(x1.device)
    numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / beta)
    denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
    return -torch.mean(torch.log(numerators / denominators), dim=0)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.beta = kwargs.get("beta", 0.08)

    def forward(self, x1, x2):
        return contrastive_loss(x1, x2, self.beta)


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


class KFCContrastiveLoss(ContrastiveLoss):

    def forward(self, x1, x2, races, bias_map):
        """
        Args:
            inputs: (x1, x2, kinship, bias_map)

        Returns:
            loss: torch.Tensor
            margin_list: list
        """
        batch_size = x1.size(0)
        AA_num = 0
        A_num = 0
        C_num = 0
        I_num = 0
        AA_idx = 0
        A_idx = 0
        C_idx = 0
        I_idx = 0
        x1x2 = torch.cat([x1, x2], dim=0)
        x2x1 = torch.cat([x2, x1], dim=0)

        cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1), torch.unsqueeze(x1x2, dim=0), dim=2) / (
            self.beta
        )
        mask = (1.0 - torch.eye(2 * x1.size(0))).to(x1.device)
        diagonal_cosine = torch.cosine_similarity(x1x2, x2x1, dim=1)

        debais_margin = torch.sum(bias_map, axis=1) / len(bias_map)
        for i in range(batch_size):
            if races[i] == 0:
                AA_num += debais_margin[i] + debais_margin[i + batch_size]
                AA_idx += 2
            elif races[i] == 1:
                A_num += debais_margin[i] + debais_margin[i + batch_size]
                A_idx += 2
            elif races[i] == 2:
                C_num += debais_margin[i] + debais_margin[i + batch_size]
                C_idx += 2
            elif races[i] == 3:
                I_num += debais_margin[i] + debais_margin[i + batch_size]
                I_idx += 2
        if AA_idx == 0:
            AA_margin = 0
        else:
            AA_margin = AA_num / AA_idx
        if A_idx == 0:
            A_margin = 0
        else:
            A_margin = A_num / A_idx
        if C_idx == 0:
            C_margin = 0
        else:
            C_margin = C_num / C_idx
        if I_idx == 0:
            I_margin = 0
        else:
            I_margin = I_num / I_idx
        numerators = torch.exp((diagonal_cosine - debais_margin) / (self.beta))
        denominators = (
            torch.sum(torch.exp(cosine_mat) * mask, dim=1) - torch.exp(diagonal_cosine / self.beta) + numerators
        )  # - x1x2/beta+ x1x2/beta+epsilon

        return -torch.mean(torch.log(numerators) - torch.log(denominators), dim=0), [
            AA_margin,
            A_margin,
            C_margin,
            I_margin,
        ]


class KFCContrastiveLossV2(ContrastiveLoss):

    def forward(self, x1, x2, races, bias_map):
        """
        Compute modified contrastive loss considering bias and kinship.

        Args:
            inputs (tuple of torch.Tensor): (x1, x2, kinship, bias_map, bias)
                - x1, x2 (torch.Tensor): Input feature vectors (batch_size, feature_dim)
                - race (torch.Tensor): Race labels (batch_size,)
                - bias_map (torch.Tensor): Map of bias values (batch_size, some_dim)

        Returns:
            torch.Tensor: The computed loss.
            list: Margins per race category.
        """
        batch_size = x1.size(0)
        x1x2 = torch.cat([x1, x2], dim=0)
        x2x1 = torch.cat([x2, x1], dim=0)
        cosine_mat = F.cosine_similarity(torch.unsqueeze(x1x2, dim=1), torch.unsqueeze(x1x2, dim=0), dim=2) / self.beta
        mask = 1.0 - torch.eye(2 * batch_size, device=x1.device)

        diagonal_cosine = F.cosine_similarity(x1x2, x2x1, dim=1)
        bias_margin = torch.sum(bias_map, dim=1) / bias_map.size(1)

        race_counts = torch.zeros(4, device=x1.device)
        race_totals = torch.zeros(4, device=x1.device)

        for i in range(4):  # Assuming race labels are {0, 1, 2, 3}
            indices = torch.cat([races == i, races == i])  # Repeat for the doubled size
            if indices.any():
                race_totals[i] = torch.sum(bias_margin[indices])
                race_counts[i] = indices.sum()

        margins = torch.where(race_counts > 0, race_totals / race_counts, torch.zeros_like(race_totals))

        numerators = torch.exp((diagonal_cosine - bias_margin) / self.beta)
        denominators = (
            torch.sum(torch.exp(cosine_mat) * mask, dim=1) - torch.exp(diagonal_cosine / self.beta) + numerators
        )

        loss = -torch.mean(torch.log(numerators / denominators))
        return loss, margins.tolist()  # AA, A, C, I
