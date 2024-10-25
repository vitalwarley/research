import torch
import torch.nn.functional as F


# Paste the code for both implementations here
# V1 Implementation
class HardContrastiveLossV6_V1(torch.nn.Module):
    # ... (paste the entire V1 implementation here)
    """
    HCL with negative pairs selection based on the alpha quantile and Feature Transformation.
    """

    def __init__(self, tau=0.2, alpha=0.8, gamma_ex=2.0, gamma_in=1.6, dim_mixing=False, normalize=False):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.gamma_ex = gamma_ex
        self.gamma_in = gamma_in
        self.dim_mixing = dim_mixing
        self.normalize = normalize

    def forward(self, embeddings, positive_pairs, stage):
        """
        Compute the contrastive loss term using both original and hard embeddings.

        Args:
            embeddings (torch.Tensor): The embeddings of the batch, shape (batch_size, embedding_dim)
            positive_pairs (list of tuples): List of tuples indicating positive pairs indices.

        Returns:
            torch.Tensor: The contrastive loss term.
        """
        _, negative_pairs = generate_pairs(embeddings, positive_pairs)
        # Generate hard embeddings and pairs
        hard_pos_embeddings = extrapolate_positive_pairs(
            embeddings, positive_pairs, self.gamma_ex, self.dim_mixing, self.normalize
        )
        # Interpolate negative pairs
        hard_neg_embeddings = interpolate_negative_pairs(embeddings, negative_pairs, self.gamma_in)

        # Calculate the similarity matrices
        original_sim = compute_similarity_matrix(embeddings)
        hard_pos_sim = compute_similarity_matrix(hard_pos_embeddings)
        hard_neg_sim = compute_similarity_matrix(hard_neg_embeddings)

        # Replace positive pairs with hard pairs
        original_sim[positive_pairs[:, 0], positive_pairs[:, 1]] = hard_pos_sim[
            positive_pairs[:, 0], positive_pairs[:, 1]
        ]
        original_sim[positive_pairs[:, 1], positive_pairs[:, 0]] = hard_pos_sim[
            positive_pairs[:, 1], positive_pairs[:, 0]
        ]
        # Replace negative pairs with hard pairs
        original_sim[negative_pairs[:, 0], negative_pairs[:, 1]] = hard_neg_sim[
            negative_pairs[:, 0], negative_pairs[:, 1]
        ]
        original_sim[negative_pairs[:, 1], negative_pairs[:, 0]] = hard_neg_sim[
            negative_pairs[:, 1], negative_pairs[:, 0]
        ]

        # Calculate losses for original embeddings
        loss_original = self.calculate_loss(original_sim, positive_pairs, stage, embeddings.device)

        contrastive_loss = loss_original

        return positive_pairs, negative_pairs, hard_pos_embeddings, hard_neg_embeddings, original_sim, contrastive_loss

    def calculate_loss(self, sim_matrix, positive_pairs, stage, device):
        """
        Calculate the contrastive loss for given similarity matrix and pairs.

        Args:
            sim_matrix (torch.Tensor): The similarity matrix of embeddings.
            positive_pairs (list of tuples): List of positive pairs indices.
            negative_pairs (list of tuples): List of negative pairs indices.
            stage (str): Current stage of training (e.g., 'train', 'sanity_check').
            device (torch.device): The device to perform calculations on.

        Returns:
            torch.Tensor: The calculated contrastive loss.
        """
        num_pairs = len(positive_pairs)
        indices_i = torch.tensor([i for i, _ in positive_pairs], device=device)
        indices_j = torch.tensor([j for _, j in positive_pairs], device=device)

        # Create masks to exclude self-similarities and positive pairs
        batch_size = sim_matrix.size(0)
        mask = torch.eye(batch_size, device=device).bool()
        mask[indices_i, indices_j] = True
        mask[indices_j, indices_i] = True

        exp_sim = torch.exp(sim_matrix / self.tau)
        exp_sim_masked = exp_sim.masked_fill(mask, 0)

        pos_sim_ij = exp_sim[indices_i, indices_j]
        pos_sim_ji = exp_sim[indices_j, indices_i]

        exp_i = exp_sim_masked[indices_i]
        exp_j = exp_sim_masked[indices_j]

        if stage in ["train", "sanity_check"]:
            # Select hard negatives based on the alpha quantile
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
        contrastive_loss /= 2 * num_pairs  # Average the loss over both directions

        return contrastive_loss


def generate_pairs(embeddings, positive_pairs):
    # TODO: replace embeddings for its size
    batch_size = embeddings.size(0)
    all_pairs = set((i, j) for i in range(batch_size) for j in range(batch_size) if i != j)
    positive_pairs_set = set((i.item(), j.item()) for i, j in positive_pairs)
    negative_pairs = list(all_pairs - positive_pairs_set)
    return torch.tensor(list(positive_pairs_set)), torch.tensor(negative_pairs)


def extrapolate_positive_pairs(embeddings, positive_pairs, alpha=2.0, dim_mixing=False, normalize=False):
    hard_pos_embeddings = torch.zeros_like(embeddings)
    # Sample 1 lambda
    if dim_mixing:
        # Sample a vector of lambdas
        lambda_ = torch.distributions.Beta(alpha, alpha).sample((embeddings.size(1),)).to(embeddings.device) + 1
    else:
        lambda_ = 1.5
    for i, j in positive_pairs:
        new_embedding_i = lambda_ * embeddings[i] + (1 - lambda_) * embeddings[j]
        new_embedding_j = lambda_ * embeddings[j] + (1 - lambda_) * embeddings[i]
        # new_embedding_i1 = alpha * embeddings[i] - (alpha - 1) * embeddings[j]
        # new_embedding_j1 = alpha * embeddings[j] - (alpha - 1) * embeddings[i]
        # new_embedding_j2 = (1 - alpha) * embeddings[j] + alpha * embeddings[i]
        if normalize:
            new_embedding_i = F.normalize(new_embedding_i, p=2, dim=0)
            new_embedding_j = F.normalize(new_embedding_j, p=2, dim=0)
        hard_pos_embeddings[i] = new_embedding_i
        hard_pos_embeddings[j] = new_embedding_j
        # hard_positive_pairs.append((new_embedding_i2, new_embedding_j2))
    return hard_pos_embeddings


def interpolate_negative_pairs(embeddings, negative_pairs, alpha=0.5):
    hard_negative_embeddings = torch.zeros_like(embeddings)
    lambda_ = 0.5
    for i, j in negative_pairs:
        new_embedding_i = lambda_ * embeddings[i] + (1 - lambda_) * embeddings[j]
        new_embedding_j = lambda_ * embeddings[j] + (1 - lambda_) * embeddings[i]
        hard_negative_embeddings[i] = new_embedding_i
        hard_negative_embeddings[j] = new_embedding_j
    return hard_negative_embeddings


def compute_similarity_matrix(embeddings):
    cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    return cosine_sim


# V2 Implementation
class HardContrastiveLossV6_V2(torch.nn.Module):
    # ... (paste the entire V2 implementation here)
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
        original_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        negative_pairs = self.generate_negative_pairs(batch_size, positive_pairs)

        # To disable the feature transformation, set gamma_ex and gamma_in to 0
        if self.gamma_ex or self.gamma_in:
            if self.gamma_ex:
                hard_pos_embeddings = self.extrapolate_positive_pairs(embeddings, positive_pairs)
                hard_pos_sim = F.cosine_similarity(
                    hard_pos_embeddings.unsqueeze(1), hard_pos_embeddings.unsqueeze(0), dim=2
                )
                original_sim[positive_pairs[:, 0], positive_pairs[:, 1]] = hard_pos_sim[
                    positive_pairs[:, 0], positive_pairs[:, 1]
                ]
                original_sim[positive_pairs[:, 1], positive_pairs[:, 0]] = hard_pos_sim[
                    positive_pairs[:, 1], positive_pairs[:, 0]
                ]
            if self.gamma_in:
                hard_neg_embeddings = self.interpolate_negative_pairs(embeddings, negative_pairs)
                hard_neg_sim = F.cosine_similarity(
                    hard_neg_embeddings.unsqueeze(1), hard_neg_embeddings.unsqueeze(0), dim=2
                )
                original_sim[negative_pairs[:, 0], negative_pairs[:, 1]] = hard_neg_sim[
                    negative_pairs[:, 0], negative_pairs[:, 1]
                ]
                original_sim[negative_pairs[:, 1], negative_pairs[:, 0]] = hard_neg_sim[
                    negative_pairs[:, 1], negative_pairs[:, 0]
                ]

        return (
            positive_pairs,
            negative_pairs,
            hard_pos_embeddings,
            hard_neg_embeddings,
            original_sim,
            self.compute_loss(original_sim, positive_pairs, stage, device),
        )

    def compute_loss(self, sim_matrix, positive_pairs, stage, device):
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

    @staticmethod
    def generate_negative_pairs(batch_size, positive_pairs):
        # TODO: implement this method performance
        all_pairs = set((i, j) for i in range(batch_size) for j in range(batch_size) if i != j)
        positive_pairs_set = set((i.item(), j.item()) for i, j in positive_pairs)
        negative_pairs = list(all_pairs - positive_pairs_set)
        return torch.tensor(negative_pairs)

    def extrapolate_positive_pairs(self, embeddings, positive_pairs):
        hard_pos_embeddings = embeddings.clone()
        lambda_ = (
            self.generate_lambda(self.gamma_ex, embeddings.size(1) if self.dim_mixing else 1, embeddings.device) + 1
        )

        i, j = positive_pairs.T
        new_embedding_i = lambda_ * embeddings[i] + (1 - lambda_) * embeddings[j]
        new_embedding_j = lambda_ * embeddings[j] + (1 - lambda_) * embeddings[i]

        if self.normalize:
            new_embedding_i = F.normalize(new_embedding_i, p=2, dim=1)
            new_embedding_j = F.normalize(new_embedding_j, p=2, dim=1)

        hard_pos_embeddings[i] = new_embedding_i
        hard_pos_embeddings[j] = new_embedding_j

        print(
            torch.cosine_similarity(embeddings[i], embeddings[j]),
            torch.cosine_similarity(new_embedding_i, new_embedding_j),
        )
        print(
            torch.cosine_similarity(embeddings[j], embeddings[i])
            >= torch.cosine_similarity(new_embedding_j, new_embedding_i)
        )
        assert all(
            torch.cosine_similarity(embeddings[i], embeddings[j])
            >= torch.cosine_similarity(new_embedding_i, new_embedding_j)
        )
        return hard_pos_embeddings

    def interpolate_negative_pairs(self, embeddings, negative_pairs):
        hard_negative_embeddings = embeddings.clone()
        lambda_ = self.generate_lambda(self.gamma_in, embeddings.size(1) if self.dim_mixing else 1, embeddings.device)

        i, j = negative_pairs.T
        new_embedding_i = lambda_ * embeddings[i] + (1 - lambda_) * embeddings[j]
        new_embedding_j = lambda_ * embeddings[j] + (1 - lambda_) * embeddings[i]

        hard_negative_embeddings[i] = new_embedding_i
        hard_negative_embeddings[j] = new_embedding_j
        # Assert similarities on hard negative pairs are higher than original
        assert all(
            torch.cosine_similarity(embeddings[i], embeddings[j])
            <= torch.cosine_similarity(new_embedding_i, new_embedding_j)
        )
        return hard_negative_embeddings

    def generate_lambda(self, value, size, device):
        return 0.5


def generate_sample_data(batch_size, embedding_dim):
    embeddings = torch.randn(batch_size, embedding_dim)
    positive_pairs = torch.randint(0, batch_size, (batch_size // 2, 2))
    return embeddings, positive_pairs


def compare_implementations():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    batch_size = 8
    embedding_dim = 128
    alpha = 0.8
    dim_mixing = False
    gamma_ex = 2.0
    gamma_in = 1.5
    tau = 0.2

    # Initialize both implementations
    v1_loss = HardContrastiveLossV6_V1(
        tau=tau, alpha=alpha, gamma_ex=gamma_ex, gamma_in=gamma_in, dim_mixing=dim_mixing
    )
    v2_loss = HardContrastiveLossV6_V2(
        tau=tau, alpha=alpha, gamma_ex=gamma_ex, gamma_in=gamma_in, dim_mixing=dim_mixing
    )

    # Generate sample data
    embeddings, positive_pairs = generate_sample_data(batch_size, embedding_dim)

    # Compute losses
    pp_v1, np_v1, hpe_v1, hne_v1, os_v1, v1_output = v1_loss(embeddings, positive_pairs, "train")
    pp_v2, np_v2, hpe_v2, hne_v2, os_v2, v2_output = v2_loss(embeddings, positive_pairs, "train")

    print(f"V1 output: {v1_output.item()}")
    print(f"V2 output: {v2_output.item()}")
    print(f"Difference: {abs(v1_output - v2_output).item()}")

    # Compare original similarity matrices
    print(f"Similarities difference: {torch.max(torch.abs(os_v1 - os_v2)).item()}")

    # Compare positive and negative pairs
    print(f"Positive pairs difference: {torch.max(torch.abs(pp_v1 - pp_v2)).item()}")
    print(f"Negative pairs difference: {torch.max(torch.abs(np_v1 - np_v2)).item()}")


if __name__ == "__main__":
    compare_implementations()
