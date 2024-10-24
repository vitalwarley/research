import torch
import torch.nn.functional as F
from scl import contrastive_loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.beta = kwargs.get("beta", 0.08)

    def forward(self, x1, x2):
        return contrastive_loss(x1, x2, self.beta)


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
