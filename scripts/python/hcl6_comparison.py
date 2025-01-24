import time

import torch
import torch.nn.functional as F


# Original Implementation
class OriginalHardContrastiveLossV6(torch.nn.Module):
    def __init__(self, tau=0.2, alpha=0.8, gamma_ex=2.0, gamma_in=1.6, dim_mixing=False, normalize=False):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.gamma_ex = gamma_ex
        self.gamma_in = gamma_in
        self.dim_mixing = dim_mixing
        self.normalize = normalize

    def forward(self, embeddings, positive_pairs, stage):
        positive_pairs, negative_pairs = self.generate_pairs(embeddings, positive_pairs)
        # print(f"Positive pairs: {positive_pairs}")
        # print(f"Negative pairs: {negative_pairs}")
        hard_pos_embeddings = self.extrapolate_positive_pairs(
            embeddings, positive_pairs, self.gamma_ex, self.dim_mixing, self.normalize
        )
        hard_neg_embeddings = self.interpolate_negative_pairs(embeddings, negative_pairs, self.gamma_in)

        original_sim = self.compute_similarity_matrix(embeddings)
        hard_pos_sim = self.compute_similarity_matrix(hard_pos_embeddings)
        hard_neg_sim = self.compute_similarity_matrix(hard_neg_embeddings)

        original_sim[positive_pairs[:, 0], positive_pairs[:, 1]] = hard_pos_sim[
            positive_pairs[:, 0], positive_pairs[:, 1]
        ]
        original_sim[positive_pairs[:, 1], positive_pairs[:, 0]] = hard_pos_sim[
            positive_pairs[:, 1], positive_pairs[:, 0]
        ]
        original_sim[negative_pairs[:, 0], negative_pairs[:, 1]] = hard_neg_sim[
            negative_pairs[:, 0], negative_pairs[:, 1]
        ]
        original_sim[negative_pairs[:, 1], negative_pairs[:, 0]] = hard_neg_sim[
            negative_pairs[:, 1], negative_pairs[:, 0]
        ]

        return self.calculate_loss(original_sim, positive_pairs, stage, embeddings.device)

    def generate_pairs(self, embeddings, positive_pairs):
        batch_size = embeddings.size(0)
        all_pairs = set((i, j) for i in range(batch_size) for j in range(batch_size) if i != j)
        positive_pairs_set = set((i.item(), j.item()) for i, j in positive_pairs)
        negative_pairs = list(all_pairs - positive_pairs_set)
        return torch.tensor(list(positive_pairs_set)), torch.tensor(negative_pairs)

    def extrapolate_positive_pairs(self, embeddings, positive_pairs, alpha=2.0, dim_mixing=False, normalize=False):
        hard_pos_embeddings = torch.zeros_like(embeddings)
        # if dim_mixing:
        #    lambda_ = torch.distributions.Beta(alpha, alpha).sample((embeddings.size(1),)).to(embeddings.device) + 1
        # else:
        #    lambda_ = torch.distributions.Beta(alpha, alpha).sample().item() + 1
        lambda_ = torch.tensor(1.5)
        for i, j in positive_pairs:
            new_embedding_i = lambda_ * embeddings[i] + (1 - lambda_) * embeddings[j]
            new_embedding_j = lambda_ * embeddings[j] + (1 - lambda_) * embeddings[i]
            if normalize:
                new_embedding_i = F.normalize(new_embedding_i, p=2, dim=0)
                new_embedding_j = F.normalize(new_embedding_j, p=2, dim=0)
            hard_pos_embeddings[i] = new_embedding_i
            hard_pos_embeddings[j] = new_embedding_j
        return hard_pos_embeddings

    def interpolate_negative_pairs(self, embeddings, negative_pairs, alpha=0.5):
        hard_negative_embeddings = torch.zeros_like(embeddings)
        # lambda_ = torch.distributions.Beta(alpha, alpha).sample().item()
        lambda_ = torch.tensor(0.5)
        for i, j in negative_pairs:
            new_embedding_i = lambda_ * embeddings[i] + (1 - lambda_) * embeddings[j]
            new_embedding_j = lambda_ * embeddings[j] + (1 - lambda_) * embeddings[i]
            hard_negative_embeddings[i] = new_embedding_i
            hard_negative_embeddings[j] = new_embedding_j
        return hard_negative_embeddings

    def compute_similarity_matrix(self, embeddings):
        return F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

    def calculate_loss(self, sim_matrix, positive_pairs, stage, device):
        num_pairs = len(positive_pairs)
        indices_i = torch.tensor([i for i, _ in positive_pairs], device=device)
        indices_j = torch.tensor([j for _, j in positive_pairs], device=device)

        mask = torch.eye(sim_matrix.size(0), device=device).bool()
        mask[indices_i, indices_j] = True
        mask[indices_j, indices_i] = True

        exp_sim = torch.exp(sim_matrix / self.tau)
        exp_sim_masked = exp_sim.masked_fill(mask, 0)

        pos_sim_ij = exp_sim[indices_i, indices_j]
        pos_sim_ji = exp_sim[indices_j, indices_i]

        exp_i = exp_sim_masked[indices_i]
        exp_j = exp_sim_masked[indices_j]

        if stage in ["train", "sanity_check"]:
            threshold_i = torch.quantile(exp_i, self.alpha, dim=1, keepdim=True)
            threshold_j = torch.quantile(exp_j, self.alpha, dim=1, keepdim=True)

            hard_neg_sims_i = torch.where(exp_i >= threshold_i, exp_i, torch.tensor(0.0, device=device))
            hard_neg_sims_j = torch.where(exp_j >= threshold_j, exp_j, torch.tensor(0.0, device=device))

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
        contrastive_loss /= 2 * num_pairs

        return contrastive_loss


# Refactored Implementation
class RefactoredHardContrastiveLossV6(torch.nn.Module):
    def __init__(self, tau=0.2, alpha=0.8, gamma_ex=2.0, gamma_in=1.6, dim_mixing=False, normalize=False):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.gamma_ex = gamma_ex
        self.gamma_in = gamma_in
        self.dim_mixing = dim_mixing
        self.normalize = normalize

    def forward(self, embeddings, positive_pairs, stage):
        device = embeddings.device
        batch_size = embeddings.size(0)

        positive_pairs = positive_pairs.clone().detach().to(device)
        negative_pairs = self.generate_negative_pairs(batch_size, positive_pairs)

        # print(f"Positive pairs: {positive_pairs}")
        # print(f"Negative pairs: {negative_pairs}")

        hard_pos_embeddings = self.extrapolate_positive_pairs(embeddings, positive_pairs)
        hard_neg_embeddings = self.interpolate_negative_pairs(embeddings, negative_pairs)

        original_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        hard_pos_sim = F.cosine_similarity(hard_pos_embeddings.unsqueeze(1), hard_pos_embeddings.unsqueeze(0), dim=2)
        hard_neg_sim = F.cosine_similarity(hard_neg_embeddings.unsqueeze(1), hard_neg_embeddings.unsqueeze(0), dim=2)

        original_sim[positive_pairs[:, 0], positive_pairs[:, 1]] = hard_pos_sim[
            positive_pairs[:, 0], positive_pairs[:, 1]
        ]
        original_sim[positive_pairs[:, 1], positive_pairs[:, 0]] = hard_pos_sim[
            positive_pairs[:, 1], positive_pairs[:, 0]
        ]
        original_sim[negative_pairs[:, 0], negative_pairs[:, 1]] = hard_neg_sim[
            negative_pairs[:, 0], negative_pairs[:, 1]
        ]
        original_sim[negative_pairs[:, 1], negative_pairs[:, 0]] = hard_neg_sim[
            negative_pairs[:, 1], negative_pairs[:, 0]
        ]

        return self.calculate_loss(original_sim, positive_pairs, stage, device)

    @staticmethod
    def generate_negative_pairs_(batch_size, positive_pairs):
        # Create all possible pairs (i, j) where i != j
        indices = torch.arange(batch_size)
        all_pairs = torch.cartesian_prod(indices, indices)
        mask = all_pairs[:, 0] != all_pairs[:, 1]
        all_pairs = all_pairs[mask]

        # Convert positive pairs into a tensor format for comparison
        positive_pairs_tensor = torch.stack([positive_pairs[:, 0], positive_pairs[:, 1]], dim=1)

        # Identify the negative pairs by filtering out positive pairs
        is_positive = (all_pairs.unsqueeze(1) == positive_pairs_tensor).all(-1).any(1)
        negative_pairs = all_pairs[~is_positive]

        return negative_pairs

    @staticmethod
    def generate_negative_pairs(batch_size, positive_pairs):
        all_pairs = set((i, j) for i in range(batch_size) for j in range(batch_size) if i != j)
        positive_pairs_set = set((i.item(), j.item()) for i, j in positive_pairs)
        negative_pairs = list(all_pairs - positive_pairs_set)
        return torch.tensor(negative_pairs)

    def extrapolate_positive_pairs(self, embeddings, positive_pairs):
        hard_pos_embeddings = embeddings.clone()
        lambda_ = self.generate_lambda(embeddings.size(1) if self.dim_mixing else 1, embeddings.device) + 1

        i, j = positive_pairs.T
        new_embedding_i = lambda_ * embeddings[i] + (1 - lambda_) * embeddings[j]
        new_embedding_j = lambda_ * embeddings[j] + (1 - lambda_) * embeddings[i]

        if self.normalize:
            new_embedding_i = F.normalize(new_embedding_i, p=2, dim=1)
            new_embedding_j = F.normalize(new_embedding_j, p=2, dim=1)

        hard_pos_embeddings[i] = new_embedding_i
        hard_pos_embeddings[j] = new_embedding_j
        return hard_pos_embeddings

    def interpolate_negative_pairs(self, embeddings, negative_pairs):
        hard_negative_embeddings = embeddings.clone()
        lambda_ = self.generate_lambda(1, embeddings.device).item()

        i, j = negative_pairs.T
        new_embedding_i = lambda_ * embeddings[i] + (1 - lambda_) * embeddings[j]
        new_embedding_j = lambda_ * embeddings[j] + (1 - lambda_) * embeddings[i]

        hard_negative_embeddings[i] = new_embedding_i
        hard_negative_embeddings[j] = new_embedding_j
        return hard_negative_embeddings

    def generate_lambda(self, size, device):
        # return torch.distributions.Beta(self.gamma_ex, self.gamma_ex).sample((size,)).to(device) + 1
        return torch.tensor(0.5)

    def calculate_loss(self, sim_matrix, positive_pairs, stage, device):
        num_pairs = len(positive_pairs)
        indices_i, indices_j = positive_pairs.T

        mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
        mask[indices_i, indices_j] = True
        mask[indices_j, indices_i] = True

        exp_sim = torch.exp(sim_matrix / self.tau)
        exp_sim_masked = exp_sim.masked_fill(mask, 0)

        pos_sim_ij = exp_sim[indices_i, indices_j]
        pos_sim_ji = exp_sim[indices_j, indices_i]

        exp_i = exp_sim_masked[indices_i]
        exp_j = exp_sim_masked[indices_j]

        if stage in ["train", "sanity_check"]:
            threshold_i = torch.quantile(exp_i, self.alpha, dim=1, keepdim=True)
            threshold_j = torch.quantile(exp_j, self.alpha, dim=1, keepdim=True)

            hard_neg_sims_i = torch.where(exp_i >= threshold_i, exp_i, torch.tensor(0.0, device=device))
            hard_neg_sims_j = torch.where(exp_j >= threshold_j, exp_j, torch.tensor(0.0, device=device))

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
        return contrastive_loss / (2 * num_pairs)


# Test sample
def test_hcl_v6():
    torch.manual_seed(42)  # For reproducibility
    batch_size = 20
    embedding_dim = 512
    num_positive_pairs = 10

    # Generate random embeddings and positive pairs
    embeddings = torch.randn(batch_size, embedding_dim)
    positive_pairs = torch.randint(0, batch_size, (num_positive_pairs, 2))

    # Initialize both implementations
    original_hcl = OriginalHardContrastiveLossV6()
    refactored_hcl = RefactoredHardContrastiveLossV6()

    # Test for both 'train' and 'test' stages
    for stage in ["train", "test"]:
        print(f"Stage: {stage}")
        # Compute losse
        print("Original Implementation")
        start_time = time.time()
        original_loss = original_hcl(embeddings, positive_pairs, stage)
        original_time = time.time() - start_time
        print("Refactored Implementation")

        start_time = time.time()
        refactored_loss = refactored_hcl(embeddings, positive_pairs, stage)
        refactored_time = time.time() - start_time

        # Compare results
        print(f"Stage: {stage}")
        print(f"Original Loss: {original_loss.item():.6f}")
        print(f"Refactored Loss: {refactored_loss.item():.6f}")
        print(f"Absolute Difference: {abs(original_loss - refactored_loss).item():.6f}")
        print(f"Original Time: {original_time:.6f} seconds")
        print(f"Refactored Time: {refactored_time:.6f} seconds")
        print(f"Speedup: {original_time / refactored_time:.2f}x")
        print()


def test_performance(batch_sizes=[8, 16, 32, 64, 128, 256, 512]):
    embedding_dim = 128
    num_positive_pairs = 8
    num_runs = 10

    original_hcl = OriginalHardContrastiveLossV6()
    refactored_hcl = RefactoredHardContrastiveLossV6()

    for batch_size in batch_sizes:
        original_times = []
        refactored_times = []

        for _ in range(num_runs):
            embeddings = torch.randn(batch_size, embedding_dim)
            positive_pairs = torch.randint(0, batch_size, (num_positive_pairs, 2))

            start_time = time.time()
            _ = original_hcl(embeddings, positive_pairs, "train")
            original_times.append(time.time() - start_time)

            start_time = time.time()
            _ = refactored_hcl(embeddings, positive_pairs, "train")
            refactored_times.append(time.time() - start_time)

        avg_original_time = sum(original_times) / num_runs
        avg_refactored_time = sum(refactored_times) / num_runs

        print(f"Batch size: {batch_size}")
        print(f"Average Original Time: {avg_original_time:.6f} seconds")
        print(f"Average Refactored Time: {avg_refactored_time:.6f} seconds")
        print(f"Speedup: {avg_original_time / avg_refactored_time:.2f}x")
        print()


if __name__ == "__main__":
    print("Running correctness test...")
    test_hcl_v6()

    # print("Running performance test...")
    # test_performance()
