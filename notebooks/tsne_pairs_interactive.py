# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Using rfiw2021 venv.

# %% [markdown]
# # Import libraries

# %%
"""
Adapted from Track1/find.py
"""
import sys
from pathlib import Path

import numpy as np

# %%
import plotly.graph_objs as go
import torch
from cuml.manifold import TSNE
from sklearn.manifold import TSNE as SKTSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
try:
    IS_NOTEBOOK = True
    HERE = Path(__file__).resolve().parent
except NameError:
    IS_NOTEBOOK = False
    HERE = Path().resolve()


# %% [markdown]
# ## Define seed


# %%
def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


seed_everything(42)

# %% [markdown]
# ## Adjust paths

# %%
print(HERE)
sys.path.insert(0, str(Path(HERE, "..")))  # kinship root
sys.path.insert(0, str(Path(HERE, "..", "rfiw2021")))  # rfiw2021 dir

# %% [markdown]
# ## Import dataset and model

# %%
from dataset import FIWPair  # noqa: E402
from Track1.models import Net  # noqa: E402

# %% [markdown]
# ## Define core functions


# %%
def compute_embeddings(face1, face2, model):
    with torch.no_grad():
        embeddings1 = model.encoder(face1.cuda()).cpu().numpy()
        embeddings2 = model.encoder(face2.cuda()).cpu().numpy()
    return embeddings1, embeddings2


def stack_embeddings_and_labels(embeddings1, embeddings2, kin_relations, face1_fids, face2_fids):
    embeddings1 = np.vstack(embeddings1)
    embeddings2 = np.vstack(embeddings2)
    kin_relations = np.vstack(kin_relations)
    face1_fids = np.vstack(face1_fids)
    face2_fids = np.vstack(face2_fids)
    return embeddings1, embeddings2, kin_relations, face1_fids, face2_fids


def fuse_embeddings(embeddings1, embeddings2, fusion):
    if fusion == "mean":
        return np.mean([embeddings1, embeddings2], axis=0)
    elif fusion == "concat":
        return np.concatenate([embeddings1, embeddings2], axis=1)
    else:
        raise ValueError('Invalid fusion method. Choose either "mean" or "concat".')


def process_data(val_loader, model):
    embeddings1_list = []
    embeddings2_list = []
    kin_relations = []
    face1_fids = []
    face2_fids = []

    for face1, face2, kin_relation, face1_fid, face2_fid, _ in tqdm(val_loader):
        embeddings1, embeddings2 = compute_embeddings(face1, face2, model)
        embeddings1_list.append(embeddings1)
        embeddings2_list.append(embeddings2)
        kin_relations.extend(kin_relation)
        face1_fids.extend(face1_fid.cpu().numpy())
        face2_fids.extend(face2_fid.cpu().numpy())

    embeddings1, embeddings2, kin_relations, face1_fids, face2_fids = stack_embeddings_and_labels(
        embeddings1_list, embeddings2_list, kin_relations, face1_fids, face2_fids
    )
    return embeddings1, embeddings2, (kin_relations, face1_fids, face2_fids)


# %%
def select_items(labels, items, index):
    print(f"Selecting {items} from face1.")
    items_unique = np.unique(labels[index])
    if isinstance(items, int):
        items = np.random.choice(items_unique, size=items, replace=False)
    print(f"Items selected from face1: {items}")
    mask = np.in1d(labels[index], items)
    indexes = np.where(mask)[0]
    print(f"Selected {len(indexes)} embeddings.")
    return indexes


def setup_embeddings(data, n_samples=1000, fids=0, kinships=0):
    print("Setting up data...")
    embeddings, labels = data
    indexes = np.arange(len(embeddings))

    if fids:
        indexes_fids = select_items(labels, fids, 1)
        indexes = np.intersect1d(indexes, indexes_fids)
    if kinships:
        indexes_krs = select_items(labels, kinships, 0)
        indexes = np.intersect1d(indexes, indexes_krs)
    if n_samples and not (fids or kinships):
        print(f"Selecting {n_samples} random from embeddings.")
        indexes = np.random.choice(indexes, size=n_samples, replace=False)

    X = embeddings[indexes]
    labels = [label[indexes] for label in labels]

    for i in range(1, 3):  # i=1 for Face1, i=2 for Face2
        face_families = labels[i]
        unique_families = np.unique(face_families)
        print(f"Selected families for Face{i}: {len(unique_families)}")

    # Assuming kinship_relations_face1 & kinship_relations_face2 are strings, let's map them to integers
    unique_kinship_relations = np.unique(labels[0])
    kinship_mapping = {relation: i for i, relation in enumerate(unique_kinship_relations)}

    is_kin = labels[1] == labels[2]
    for relation, index in kinship_mapping.items():
        n_kr_pos = sum(labels[0][is_kin] == relation)
        n_kr_neg = sum(labels[0][~is_kin] == relation)
        print(f"{relation}: {n_kr_pos} positives and {n_kr_neg} negatives")

    return X, labels


def create_tsne_model(n_samples, n_components=2, perplexity=30, early_exaggeration=12, verbose=False):
    print("Setting up t-SNE...")
    if n_components == 2:
        print(n_samples, early_exaggeration)
        lr = max(n_samples / early_exaggeration / 4, 50)
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=lr, n_iter=5000, verbose=verbose)
    elif n_components == 3:
        tsne = SKTSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate="auto",
            n_iter=5000,
            verbose=verbose,
            init="random",
        )

    return tsne


def set_marker_colors(labels):
    # Create markers based on kin_relations
    marker_colors = np.where(labels[1] == labels[2], "blue", "red").reshape(
        -1,
    )
    return marker_colors


def create_scatter_plot(tsne_results, labels, marker_colors, n_components=2):
    unique_kinship_relations = np.unique(labels[0])
    marker_types = ["circle", "square", "diamond", "cross", "x", "circle-open", "diamond-open", "square-open"]
    marker_mapping = {
        relation: marker_types[i % len(marker_types)] for i, relation in enumerate(unique_kinship_relations)
    }
    # Create separate scatter plots for each kinship relation
    scatters = []
    for kinship_relation in unique_kinship_relations:
        indexes = [i for i, relation in enumerate(labels[0]) if relation == kinship_relation]
        marker_symbol = marker_mapping[kinship_relation]
        marker_color = [marker_colors[i] for i in indexes]

        if n_components == 2:
            scatter = go.Scatter(
                x=tsne_results[indexes, 0],
                y=tsne_results[indexes, 1],
                mode="markers",
                marker=dict(color=marker_color, symbol=marker_symbol, size=5),
                name=kinship_relation,  # Set the name to kinship_relation for the legend
            )
        elif n_components == 3:
            scatter = go.Scatter3d(
                x=tsne_results[indexes, 0],
                y=tsne_results[indexes, 1],
                z=tsne_results[indexes, 2],
                mode="markers",
                marker=dict(color=marker_color, symbol=marker_symbol, size=5),
                name=kinship_relation,  # Set the name to kinship_relation for the legend
            )

        scatters.append(scatter)

    return scatters


def create_final_figure(scatters):
    fig = go.Figure(data=scatters)
    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=go.layout.Margin(
            l=0, r=0, b=0, t=0, pad=4  # left margin  # right margin  # bottom margin  # top margin  # padding
        ),
        legend=dict(
            orientation="v",  # this sets the legend orientation as horizontal
            yanchor="bottom",  # this sets the y anchor of the legend to the bottom
            y=0.00,  # this adjusts the position along y axis (move it slightly above the bottom)
            xanchor="right",  # this sets the x anchor of the legend to the right
            x=1,  # this adjusts the position along x axis (put it all the way to the right)
        ),
    )
    fig.show()


def plot_tsne(data, n_samples=1000, fids=0, kinships=0, n_components=2, perplexity=30, verbose=False, filter="before"):
    embeddings, labels = data
    n_samples = None
    if filter == "before":
        # Filter embeddings
        embeddings, labels = setup_embeddings(data, n_samples, fids, kinships)
        n_samples = n_samples if not (fids or kinships) else embeddings.shape[0]
        # Create t-SNE model
        tsne = create_tsne_model(n_samples, n_components, perplexity, verbose=verbose)
        # Fit model to embeddings
        tsne_results = tsne.fit_transform(embeddings)
    elif filter == "after":
        n_samples = embeddings.shape[0]
        # Create t-SNE model
        tsne = create_tsne_model(n_samples, n_components, perplexity, verbose=verbose)
        # Fit model to embeddings
        tsne_results = tsne.fit_transform(embeddings)
        # Filter embeddings
        embeddings, labels = setup_embeddings(data, n_samples, fids, kinships)
    else:
        raise ValueError(f"filter={filter} invalid.")
    n_samples_to_plot = embeddings.shape[0]
    print(f"# embeddings to plot: {n_samples_to_plot} [filter={filter}]")
    marker_colors = set_marker_colors(labels)
    scatter = create_scatter_plot(tsne_results, labels, marker_colors, n_components)
    create_final_figure(scatter)


# %% [markdown]
# ### Define tetup data and model function


# %%
def setup_data(root_dir, csv_path, batch_size, samples_per_member: int = 0, families: list = []):
    # Loading and sampling the dataset
    val_dataset = FIWPair(
        root_dir=root_dir, csv_path=csv_path, families=families, samples_per_member=samples_per_member
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=False)
    return val_loader


def setup_model(model_path):
    # Loading model
    model = Net().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


# %% [markdown]
# ## Define main paths

# %%
root_dir = Path(HERE, "../rfiw2021/Track1/")
csv_path = Path(HERE, "../rfiw2021/Track1/sample0/val.txt")
batch_size = 40
model_path = Path(HERE, "../rfiw2021/Track1/model_track1.pth")
plot_path = ""

# %% [markdown]
# ## Load model

# %%
model = setup_model(model_path)

# %% [markdown]
# ## Load dataloader

# %%
val_loader = setup_data(root_dir, csv_path, batch_size)

# %% [markdown]
# ## Compute embeddings

# %%
# Specify the files' paths
embeddings1_path = "data/embeddings/embeddings1.npy"
embeddings2_path = "data/embeddings/embeddings2.npy"
labels_path = "data/embeddings/labels.npy"

Path("data/embeddings").mkdir(parents=True, exist_ok=True)

# Check if the files exist
if Path(embeddings1_path).exists() and Path(embeddings2_path).exists() and Path(labels_path).exists():
    # Load the data
    embeddings1 = np.load(embeddings1_path)
    embeddings2 = np.load(embeddings2_path)
    labels = np.load(labels_path)
    print("Embeddings and labels loaded.")
    labels = (labels[0].reshape(-1), labels[1].astype(int).reshape(-1), labels[2].astype(int).reshape(-1))
else:
    # Process the data
    embeddings1, embeddings2, labels = process_data(val_loader, model)  # ~10min

    # Save the data
    np.save(embeddings1_path, embeddings1)
    np.save(embeddings2_path, embeddings2)
    np.save(labels_path, labels)
    print("Embeddings and labels computed and saved.")

fused_concat_embeddings = fuse_embeddings(embeddings1, embeddings2, fusion="concat")
data = (fused_concat_embeddings, labels)

# %% [markdown]
# # t-SNE with filtering before .fit

# %% [markdown]
# ## t-SNE plotsfor 5 specific families

# %% [markdown]
# ### t-SNN plots (n_componeents = 3, perplexities in [20, 50, 100])
#
# Families are drawn from face1. That means that face2 can also be from any other family.

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(n_components=3, fids=[250, 283, 409, 735, 873], perplexity=perplexity)

# %% [markdown]
# ### t-SNN plots (n_components=2, perplexities in [20, 50, 100])
#
# Families are drawn from face1. That means that face2 can also be from any other family.

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(n_components=2, fids=[250, 283, 409, 735, 873], perplexity=perplexity)

# %% [markdown]
# ## t-SNN plots for 10 random families

# %% [markdown]
# ### t-SNN plots (n_components=3, perplexities in [20, 50, 100])
#
# Families are drawn from face1. That means that face2 can also be from any other family.

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(n_components=3, fids=10, perplexity=perplexity)

# %% [markdown]
# ### t-SNN plots (n_components=2, perplexities in [20, 50, 100])
#
# Families are drawn from face1. That means that face2 can also be from any other family.

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(n_components=2, fids=10, perplexity=perplexity)

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(n_components=3, fids=[250, 283, 409, 735, 873], kinships=["bb", "sibs", "ss"], perplexity=perplexity)

# %% [markdown]
# ## t-SNN plots for 5 specific families

# %% [markdown]
# ### t-SNN plots (n_components = 3, perplexities in [20, 50, 100])
#
# Families are drawn from face1. That means that face2 can also be from any other family.

# %% [markdown]
# #### Only bb, sibs, ss kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(n_components=3, fids=[250, 283, 409, 735, 873], kinships=["bb", "sibs", "ss"], perplexity=perplexity)

# %% [markdown]
# #### Only fd, md, fs, ms kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(n_components=3, fids=[250, 283, 409, 735, 873], kinships=["fd", "md", "fs", "ms"], perplexity=perplexity)

# %% [markdown]
# #### Only gfgd, gmgd, gfgs, gmgs kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(
        n_components=3, fids=[250, 283, 409, 735, 873], kinships=["gfgd", "gmgd", "gfgs", "gmgs"], perplexity=perplexity
    )

# %% [markdown]
# ## t-SNN plots for 5 specific families
#
# Families are drawn from face1. That means that face2 can also be from any other family.

# %% [markdown]
# ### n_components = 2, perplexities in [20, 50, 100]

# %% [markdown]
# #### Only bb, sibs, ss kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(
        data,
        n_components=2,
        fids=[250, 283, 409, 735, 873],
        kinships=["bb", "sibs", "ss"],
        perplexity=perplexity,
        verbose=True,
    )

# %% [markdown]
# #### Only fd, md, fs, ms kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(
        data, n_components=2, fids=[250, 283, 409, 735, 873], kinships=["fd", "md", "fs", "ms"], perplexity=perplexity
    )

# %% [markdown]
# #### Only gfgd, gmgd, gfgs, gmgs kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(
        data,
        n_components=2,
        fids=[250, 283, 409, 735, 873],
        kinships=["gfgd", "gmgd", "gfgs", "gmgs"],
        perplexity=perplexity,
    )

# %% [markdown]
# # SNE with filtering after .fit

# %% [markdown]
# ## t-SNN plots for 5 specific families

# %% [markdown]
# ### t-SNN plots (n_components = 3, perplexities in [20, 50, 100])
#
# Families are drawn from face1. That means that face2 can also be from any other family.

# %% [markdown]
# #### Only bb, sibs, ss kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(
        data,
        n_components=3,
        fids=[250, 283, 409, 735, 873],
        kinships=["bb", "sibs", "ss"],
        perplexity=perplexity,
        filter="after",
    )

# %% [markdown]
# #### Only fd, md, fs, ms kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(n_components=3, fids=[250, 283, 409, 735, 873], kinships=["fd", "md", "fs", "ms"], perplexity=perplexity)

# %% [markdown]
# #### Only gfgd, gmgd, gfgs, gmgs kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(
        n_components=3, fids=[250, 283, 409, 735, 873], kinships=["gfgd", "gmgd", "gfgs", "gmgs"], perplexity=perplexity
    )

# %% [markdown]
# ## t-SNN plots for 5 specific families
#
# Families are drawn from face1. That means that face2 can also be from any other family.

# %% [markdown]
# ### n_components = 2, perplexities in [20, 50, 100]

# %% [markdown]
# #### Only bb, sibs, ss kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(
        data,
        n_components=2,
        fids=[250, 283, 409, 735, 873],
        kinships=["bb", "sibs", "ss"],
        perplexity=perplexity,
        verbose=True,
        filter="after",
    )

# %% [markdown]
# #### Only fd, md, fs, ms kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(
        data,
        n_components=2,
        fids=[250, 283, 409, 735, 873],
        kinships=["fd", "md", "fs", "ms"],
        perplexity=perplexity,
        filter="after",
    )

# %% [markdown]
# #### Only gfgd, gmgd, gfgs, gmgs kinship relations

# %%
for perplexity in [20, 50, 100]:
    plot_tsne(
        data,
        n_components=2,
        fids=[250, 283, 409, 735, 873],
        kinships=["gfgd", "gmgd", "gfgs", "gmgs"],
        perplexity=perplexity,
        filter="after",
    )

# %%
