# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import argparse
import os
import sys
from pathlib import Path

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
import scienceplots
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
# %matplotlib inline

# %%
# %load_ext autoreload
# %autoreload 2



plt.style.use(["science", "ieee", "grid", "no-latex"])

# %%
try:
    IS_NOTEBOOK = True
    HERE = Path(__file__).resolve().parent
except NameError:
    IS_NOTEBOOK = False
    HERE = Path().resolve()

# %%
print(HERE)
sys.path.insert(0, str(Path(HERE, "..")))  # kinship root sys.path.insert(0, str(Path(HERE, "..")))  # kinship root

# %%
print(sys.path)

# %%
from dataset import FIW  # noqa: E402
from models.attention import FaCoRAttention
from models.facornet import FaCoRNetLightning, FaCoRV0

from ours.models.base import SimpleModel  # noqa: E402
from ours.models.scl import SCL  # noqa: E402


# %%
def extract_embeddings(val_loader, model):
    embeddings = []
    labels = []

    for img, family_id in tqdm(val_loader):
        with torch.no_grad():
            embedding, _ = model(img.cuda())
            embedding = embedding.cpu().numpy()
            embedding = normalize(embedding)
            embeddings.append(embedding)
            labels.append(family_id)

    # Now, embeddings contain all the embeddings from your model
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)

    return embeddings, labels


from collections import defaultdict

# %%
from scipy.spatial.distance import cosine


def cosine_similarity(a, b):
    """Compute the cosine similarity between two vectors."""
    return 1 - cosine(a, b)


def rank_top_families(embeddings, family_ids, n=5):
    """Rank the top 5 families by highest pairwise mean similarity.

    Args:
        embeddings (np.ndarray): 2D array of shape (n_samples, n_features).
        family_ids (list): List of family IDs corresponding to each embedding.

    Returns:
        list: Top 5 family IDs ranked by mean similarity.
    """
    family_similarity = defaultdict(list)

    # Group embeddings by family
    family_groups = defaultdict(list)
    for emb, fam_id in zip(embeddings, family_ids):
        family_groups[fam_id].append(emb)

    # Calculate pairwise similarities within each family
    for fam_id, emb_group in family_groups.items():
        num_embeddings = len(emb_group)
        if num_embeddings < 2:
            continue  # Skip families with less than 2 embeddings

        total_similarity = 0
        count = 0

        for i in range(num_embeddings):
            for j in range(i + 1, num_embeddings):
                sim = cosine_similarity(emb_group[i], emb_group[j])
                total_similarity += sim
                count += 1

        mean_similarity = total_similarity / count if count > 0 else 0
        family_similarity[fam_id] = mean_similarity

    # Sort families by mean similarity
    sorted_families = sorted(family_similarity.items(), key=lambda x: x[1], reverse=True)
    print(sorted_families[:n], sorted_families[-n:])

    # Get top 5 families
    top_five_families = [fam_id for fam_id, _ in sorted_families]
    # Calculate the step size
    num_samples = len(top_five_families)
    step = len(top_five_families) // num_samples

    # Select samples
    samples = [top_five_families[i * step] for i in range(num_samples)]

    return samples


# Example usage:
embeddings = np.array([[1, 2], [2, 3], [3, 4], [1, 0], [2, 1]])
family_ids = ["A", "A", "B", "B", "C"]
print(rank_top_families(embeddings, family_ids))

# %%
from scipy.spatial.distance import pdist, squareform


def cosine_similarity(a, b):
    """Compute the cosine similarity between two vectors."""
    return 1 - cosine(a, b)


def rank_top_clustered_families(embeddings, family_ids, n=5):
    """Rank families by lowest variance in pairwise distances.

    Args:
        embeddings (np.ndarray): 2D array of shape (n_samples, n_features).
        family_ids (list): List of family IDs corresponding to each embedding.
        n (int): Number of families to return.

    Returns:
        list: Top N family IDs ranked by lowest variance in pairwise distances.
    """
    family_variance = defaultdict(list)

    # Group embeddings by family
    family_groups = defaultdict(list)
    for emb, fam_id in zip(embeddings, family_ids):
        family_groups[fam_id].append(emb)

    # Calculate pairwise distances within each family
    for fam_id, emb_group in family_groups.items():
        num_embeddings = len(emb_group)
        if num_embeddings < 2:
            continue  # Skip families with less than 2 embeddings

        # Calculate pairwise distances using cosine distance
        distance_matrix = squareform(pdist(emb_group, metric="cosine"))

        # Flatten the distance matrix and compute variance
        distances = distance_matrix[np.triu_indices(num_embeddings, k=1)]
        variance = np.var(distances) if distances.size > 0 else 0
        family_variance[fam_id] = variance

    # Sort families by variance (lower is better)
    sorted_families = sorted(family_variance.items(), key=lambda x: x[1])
    print(sorted_families[:n], sorted_families[-n:])

    # Get top N families
    top_n_families = [fam_id for fam_id, _ in sorted_families[:n]]

    return top_n_families


# Example usage:
embeddings = np.array([[1, 2], [2, 3], [3, 4], [1, 0], [2, 1]])
family_ids = ["A", "A", "B", "B", "C"]
print(rank_top_clustered_families(embeddings, family_ids))

# %%
import random

from scipy.spatial.distance import euclidean


def compute_centroids(embeddings, family_ids):
    """Compute the centroid for each family group.

    Args:
        embeddings (np.ndarray): 2D array of shape (n_samples, n_features).
        family_ids (list): List of family IDs corresponding to each embedding.

    Returns:
        dict: A dictionary mapping family IDs to their centroids.
    """
    family_groups = defaultdict(list)

    # Group embeddings by family
    for emb, fam_id in zip(embeddings, family_ids):
        family_groups[fam_id].append(emb)

    # Calculate centroids
    centroids = {}
    for fam_id, emb_group in family_groups.items():
        centroids[fam_id] = np.mean(emb_group, axis=0)

    return centroids


def select_most_distant_families(embeddings, family_ids, n=5):
    """Select a random family and the n-1 most distant families based on centroid distances.

    Args:
        embeddings (np.ndarray): 2D array of shape (n_samples, n_features).
        family_ids (list): List of family IDs corresponding to each embedding.
        n (int): Total number of families to select (1 random family + n-1 distant).

    Returns:
        list: A list of family IDs including the random family and the most distant families.
    """
    # Compute centroids
    centroids = compute_centroids(embeddings, family_ids)

    # Select a random family
    random_family = random.choice(list(centroids.keys()))

    # Calculate distances from the centroid of the selected random family to all other centroids
    distances = {}
    random_centroid = centroids[random_family]

    for fam_id, centroid in centroids.items():
        if fam_id != random_family:  # Exclude the random family itself
            distances[fam_id] = euclidean(random_centroid, centroid)

    # Sort families by distance (descending) and select the top n-1
    most_distant_families = sorted(distances.items(), key=lambda x: x[1], reverse=True)[: n - 1]

    # Return the random family and its most distant families
    selected_families = [random_family] + [fam_id for fam_id, _ in most_distant_families]

    return selected_families


# Example usage:
embeddings = np.array([[1, 2], [2, 3], [3, 4], [1, 0], [2, 1]])
family_ids = ["A", "A", "B", "B", "C"]
print(select_most_distant_families(embeddings, family_ids))


# %%
def select_sequentially_distant_families(embeddings, family_ids, n=5):
    """Select a random family and the n-1 most distant families based on sequential distances.

    Args:
        embeddings (np.ndarray): 2D array of shape (n_samples, n_features).
        family_ids (list): List of family IDs corresponding to each embedding.
        n (int): Total number of families to select (1 random family + n-1 distant).

    Returns:
        list: A list of family IDs including the random family and the most distant families.
    """
    # Compute centroids
    centroids = compute_centroids(embeddings, family_ids)

    # Select a random family
    selected_families = []
    random_family = random.choice(list(centroids.keys()))
    selected_families.append(random_family)

    # Start with the random centroid
    current_centroid = centroids[random_family]

    for _ in range(1, n):
        # Calculate distances from the current centroid to all other centroids
        distances = {}

        for fam_id, centroid in centroids.items():
            if fam_id not in selected_families:  # Exclude already selected families
                distances[fam_id] = euclidean(current_centroid, centroid)

        # Find the most distant family
        most_distant_family = max(distances, key=distances.get)
        selected_families.append(most_distant_family)

        # Update the current centroid
        current_centroid = centroids[most_distant_family]

    return selected_families


from pathlib import Path

import matplotlib.pyplot as plt

# %%
import numpy as np
import umap
from sklearn.manifold import TSNE


def plot_embeddings(embeddings, labels, plot_path):
    # Set up perplexity values
    n_embeddings = len(embeddings)
    perplexities = [10, 30, 50]

    # Create a color map for families
    family_ids = rank_top_families(embeddings, labels, n=5)
    colors = ["red", "blue", "green", "purple", "orange"]
    color_map = dict(list(zip(family_ids, colors)))

    # Define n_subplots based on number of perplexity values
    n_subplots = len(perplexities) + 1  # +1 for UMAP
    n_col = 2
    n_row = int(np.ceil(n_subplots / n_col))

    # Prepare a figure to hold the subplots
    fig, axes = plt.subplots(n_row, n_col, figsize=(15, 5 * n_row))
    axes = axes.flatten()

    # Generate and plot t-SNE for different perplexity values
    for i, perplexity in enumerate(perplexities):
        print(f"Generating t-SNE with perplexity={perplexity}")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=100, n_iter=2000, metric="cosine", init="pca")
        embeddings_2d = tsne.fit_transform(embeddings)

        ax = axes[i]  # Get the current subplot

        # Plot
        for label, color in color_map.items():
            idxs = [idx for idx, val in enumerate(labels) if val == str(label)]
            ax.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], color=color, label=f"Family #0{label}")

        ax.set_title(f"t-SNE Perplexity: {perplexity}")
        ax.legend()

    # Generate and plot UMAP
    print("Generating UMAP...")
    umap_model = umap.UMAP(n_components=2, random_state=100, metric="cosine", n_neighbors=10)
    embeddings_umap = umap_model.fit_transform(embeddings)

    ax = axes[n_subplots - 1]  # Last subplot for UMAP
    for label, color in color_map.items():
        idxs = [idx for idx, val in enumerate(labels) if val == str(label)]
        ax.scatter(embeddings_umap[idxs, 0], embeddings_umap[idxs, 1], color=color, label=f"Family #0{label}")

    ax.set_title("UMAP")
    ax.legend()

    plt.tight_layout()
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
    plt.show()


# %%
def setup(root_dir, ckpt_path, batch_size, samples_per_member, gpu: int = 0):
    # Loading and sampling the dataset
    val_dataset = FIW(root_dir=root_dir, families=[250, 283, 409, 735, 873], samples_per_member=samples_per_member)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=False)

    # Loading model
    checkpoint = torch.load(ckpt_path)
    # simple_model = SimpleModel('adaface_ir_101')
    # model = SCL(model=simple_model, loss=None)
    facor = FaCoRV0()
    model = FaCoRNetLightning(model=facor, loss=None)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.model.backbone
    model.eval()
    model.cuda()

    torch.manual_seed(100)

    return model, val_loader


# %%
def run(model, val_loader, plot_path: str = ""):
    # Extracting embeddings
    embeddings, labels = extract_embeddings(val_loader, model)

    # Plotting
    plot_embeddings(embeddings, labels, plot_path)


# %%
def parser():
    parser = argparse.ArgumentParser(description="plot embeddings")
    parser.add_argument("--root_dir", type=str, help="root directory of dataset")
    parser.add_argument("--ckpt_path", type=str, help="model save path")
    parser.add_argument("--plot_path", type=str, help="plot save path")
    parser.add_argument("--batch_size", type=int, default=40, help="batch size default 40")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args


# %%
if IS_NOTEBOOK:
    if __name__ == "__main__":
        args = parser()
        run(**vars(args))
else:
    root_dir = Path(HERE, "../datasets/facornet/images/Validation_A/val-faces")
    batch_size = 40
    experiment = "686449944e814f7fab46150a63f521f4"
    checkpoint = "25-4.909-1.047-0.873813.ckpt"
    ckpt_path = Path(Path.home(), f".guild/runs/{experiment}/exp/checkpoints/{checkpoint}")
    plot_path = "plots_experiments/sota_tsne.png"
    model, val_loader = setup(root_dir, ckpt_path, batch_size, samples_per_member=3)
    run(model, val_loader, plot_path)

# %%

# %%
