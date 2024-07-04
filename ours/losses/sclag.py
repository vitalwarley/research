import torch
import torch.nn.functional as F


class ContrastiveLossWithAttributes(torch.nn.Module):
    def __init__(self, tau=0.1, lambda_g=1.0, lambda_a=1.0, margin=0.5):
        """
        Initialize the loss class with parameters.

        Args:
            tau (float): The temperature parameter for contrastive loss.
            lambda_g (float): Weight for the gender loss term.
            lambda_a (float): Weight for the age loss term.
        """
        super().__init__()
        self.tau = tau
        self.lambda_g = lambda_g
        self.lambda_a = lambda_a
        self.margin = margin

    def gender_loss(self, embeddings, gender_labels):
        """
        Compute the gender loss term in a vectorized manner.

        Args:
            embeddings (torch.Tensor): The embeddings of the batch, shape (batch_size, embedding_dim)
            gender_labels (torch.Tensor): The gender labels of the batch, shape (batch_size,)

        Returns:
            torch.Tensor: The gender loss term.
        """
        gender_mask = (gender_labels.unsqueeze(0) == gender_labels.unsqueeze(1)).float()
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        positive_pairs = gender_mask * cosine_sim
        negative_pairs = (1 - gender_mask) * F.relu(self.margin - cosine_sim)

        gender_loss = (1 - positive_pairs).sum() + negative_pairs.sum()
        return gender_loss / (gender_mask.sum() + (1 - gender_mask).sum())

    def age_loss(self, embeddings, age_labels):
        """
        Compute the age loss term in a vectorized manner.

        Args:
            embeddings (torch.Tensor): The embeddings of the batch, shape (batch_size, embedding_dim)
            age_labels (torch.Tensor): The age labels of the batch, shape (batch_size,)

        Returns:
            torch.Tensor: The age loss term.
        """
        age_mask = (age_labels.unsqueeze(0) == age_labels.unsqueeze(1)).float()
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        positive_pairs = age_mask * cosine_sim
        negative_pairs = (1 - age_mask) * F.relu(self.margin - cosine_sim)

        age_loss = (1 - positive_pairs).sum() + negative_pairs.sum()
        return age_loss / (age_mask.sum() + (1 - age_mask).sum())

    def contrastive_loss(self, embeddings, positive_pairs):
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

        cosine_sim_masked = cosine_sim.masked_fill(mask, -float("inf")) / self.tau
        exp_cosine_sim = torch.exp(cosine_sim / self.tau)

        contrastive_loss = 0
        num_pairs = len(positive_pairs)

        for i, j in positive_pairs:
            pos_sim_ij = cosine_sim[i, j] / self.tau
            pos_sim_ji = cosine_sim[j, i] / self.tau

            neg_sims_i = cosine_sim_masked[i]
            neg_sims_j = cosine_sim_masked[j]

            loss_ij = -torch.log(torch.exp(pos_sim_ij) / (torch.exp(pos_sim_ij) + torch.sum(torch.exp(neg_sims_i))))
            loss_ji = -torch.log(torch.exp(pos_sim_ji) / (torch.exp(pos_sim_ji) + torch.sum(torch.exp(neg_sims_j))))

            contrastive_loss += loss_ij + loss_ji

        contrastive_loss /= 2 * num_pairs  # Average the loss over both directions

        return contrastive_loss

    def forward(self, embeddings, positive_pairs, gender_labels, age_labels):
        """
        Compute the combined loss with contrastive loss, gender loss, and age loss in a vectorized manner.

        Args:
            embeddings (torch.Tensor): The embeddings of the batch, shape (batch_size, embedding_dim)
            positive_pairs (list of tuples): List of tuples indicating positive pairs indices.
            gender_labels (torch.Tensor): The gender labels of the batch, shape (batch_size,)
            age_labels (torch.Tensor): The age labels of the batch, shape (batch_size,)
            age_categories (dict): Dictionary mapping age labels to categories.

        Returns:
            torch.Tensor: The combined loss.
        """
        contrastive_loss_value = self.contrastive_loss(embeddings, positive_pairs)
        gender_loss_value = self.gender_loss(embeddings, gender_labels)
        age_loss_value = self.age_loss(embeddings, age_labels)
        total_loss = contrastive_loss_value + self.lambda_g * gender_loss_value + self.lambda_a * age_loss_value
        return total_loss, contrastive_loss_value, gender_loss_value, age_loss_value


# Example usage
if __name__ == "__main__":
    embeddings = torch.rand(32, 128)  # Example embeddings for a batch of 32 samples, each of dimension 128
    positive_pairs = [(i, (i + 1) % 32) for i in range(32)]  # Example positive pairs
    gender_labels = torch.randint(0, 2, (32,))  # Example gender labels
    age_labels = torch.randint(18, 60, (32,))  # Example age labels

    # Define age categories mapping
    age_categories = {age: age // 10 for age in range(18, 60)}  # Group ages by decades

    loss_calculator = ContrastiveLossWithAttributes(tau=0.1, lambda_g=1.0, lambda_a=1.0)
    loss = loss_calculator.combined_loss(embeddings, positive_pairs, gender_labels, age_labels, age_categories)
    print(loss)
