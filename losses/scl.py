import torch
import torch.nn.functional as F


class HCL(torch.nn.Module):
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

    def forward(self, embeddings, positive_pairs, stage):
        sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        return self.compute_loss(sim_matrix, positive_pairs, stage)

    def compute_loss(self, sim_matrix, positive_pairs, stage):
        device = sim_matrix.device
        num_pairs = len(positive_pairs)
        indices_i, indices_j = positive_pairs.T

        # Mask shape is (2 * num_pairs, 2 * num_pairs)
        # Diagonal are same-individual pairs, therefore we mask them
        mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
        mask[indices_i, indices_j] = True
        mask[indices_j, indices_i] = True

        # exp_sim shape is (2 * num_pairs, 2 * num_pairs)
        exp_sim = torch.exp(sim_matrix / self.tau)
        # Positive pairs are masked
        exp_sim_masked = exp_sim.masked_fill(mask, 0)

        # pos_sim_ij/ji shape are num_pairs in training, but in
        # validation we can't know which pairs are positive
        pos_sim_ij = exp_sim[indices_i, indices_j]
        pos_sim_ji = exp_sim[indices_j, indices_i]

        # neg_exp_i/j shape is (len(indices_i/j), num_pairs in training
        neg_exp_i = exp_sim_masked[
            indices_i
        ]  # all negative pairs for each individual in indices_i
        neg_exp_j = exp_sim_masked[
            indices_j
        ]  # all negative pairs for each individual in indices_j

        if stage in ["train", "sanity_check"]:  # pytorch lightning needs
            # threshold_i shape is (len(indices_i), 1); similarity threshold for each individual in indices_i
            threshold_i = torch.quantile(neg_exp_i, self.alpha_neg, dim=1, keepdim=True)
            # threshold_j shape is (len(indices_j), 1); similarity threshold for each individual in indices_j
            threshold_j = torch.quantile(neg_exp_j, self.alpha_neg, dim=1, keepdim=True)
            # These thresholds are used to filter out negative pairs with similarity lower than the threshold
            # The threshold is an interpolation between the the middle two values of the sorted similarity values
            # For example, for the sorted similarity values [-1.0000, -0.6834, 0.1654, 1.0000]
            # the threshold is -0.2590 - the middle two values are -0.6834 and 0.1654

            # hard_neg_sims_i shape is neg_exp_i.shape
            hard_neg_sims_i = torch.where(
                neg_exp_i >= threshold_i, neg_exp_i, torch.tensor(0.0, device=device)
            )
            # hard_neg_sims_j shape is neg_exp_j.shape
            hard_neg_sims_j = torch.where(
                neg_exp_j >= threshold_j, neg_exp_j, torch.tensor(0.0, device=device)
            )

            # sum_hard_neg_sims_i shape is len(indices_i) (or num_pairs)
            sum_hard_neg_sims_i = hard_neg_sims_i.sum(dim=1)
            # sum_hard_neg_sims_j shape is len(indices_j) (or num_pairs)
            sum_hard_neg_sims_j = hard_neg_sims_j.sum(dim=1)

            # threshold_pos_i shape is num_pairs
            threshold_pos_i = torch.quantile(pos_sim_ij, 1 - self.alpha_pos)
            # threshold_pos_j shape is num_pairs
            threshold_pos_j = torch.quantile(pos_sim_ji, 1 - self.alpha_pos)

            # pos_sim_ij/ji shape is num_pairs
            pos_sim_ij = torch.where(
                pos_sim_ij <= threshold_pos_i,
                pos_sim_ij,
                torch.tensor(0.0, device=device),
            )
            pos_sim_ji = torch.where(
                pos_sim_ji <= threshold_pos_j,
                pos_sim_ji,
                torch.tensor(0.0, device=device),
            )

            # loss_ij/ji shape is num_pairs
            loss_ij = -torch.log(
                (pos_sim_ij + self.eps) / (pos_sim_ij + sum_hard_neg_sims_i + self.eps)
            )
            loss_ji = -torch.log(
                (pos_sim_ji + self.eps) / (pos_sim_ji + sum_hard_neg_sims_j + self.eps)
            )
        else:
            sum_neg_sims_i = neg_exp_i.sum(dim=1)
            sum_neg_sims_j = neg_exp_j.sum(dim=1)

            loss_ij = -torch.log(pos_sim_ij / (pos_sim_ij + sum_neg_sims_i))
            loss_ji = -torch.log(pos_sim_ji / (pos_sim_ji + sum_neg_sims_j))

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
