# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Using rfiw2021 venv.

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
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
try:
    IS_NOTEBOOK = True
    HERE = Path(__file__).resolve().parent
except NameError:
    IS_NOTEBOOK = False
    HERE = Path().resolve()

# %%
print(HERE)
sys.path.insert(0, str(Path(HERE, "..")))  # kinship root
sys.path.insert(0, str(Path(HERE, "..", "rfiw2021")))  # rfiw2021 dir

# %%
from dataset import FIWPair  # noqa: E402
from Track1.models import Net  # noqa: E402
from Track1.utils import set_seed  # noqa: E402


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

    set_seed(100)

    return model


# %%
root_dir = Path(HERE, "../rfiw2021/Track1/")
csv_path = Path(HERE, "../rfiw2021/Track1/sample0/val.txt")
batch_size = 40
model_path = Path(HERE, "../rfiw2021/Track1/model_track1.pth")
plot_path = ""

# %%
model = setup_model(model_path)

# %%
val_loader = setup_data(root_dir, csv_path, batch_size)

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
else:
    # Process the data
    embeddings1, embeddings2, labels = process_data(val_loader, model)  # ~10min

    # Save the data
    np.save(embeddings1_path, embeddings1)
    np.save(embeddings2_path, embeddings2)
    np.save(labels_path, labels)
    print("Embeddings and labels computed and saved.")

fused_concat_embeddings = fuse_embeddings(embeddings1, embeddings2, fusion="concat")

# %%
labels[1]

# %%
data = (fused_concat_embeddings, labels)


def plot_tsne(n_samples=1000, fids=0, n_components=2, perplexity=30):
    print("Setting up data...")
    embeddings, labels = data
    indexes = None
    if fids:
        if isinstance(fids, int):
            fids_unique = np.unique(labels[1])
            fids = np.random.choice(fids_unique, size=fids, replace=False)
            mask = np.in1d(labels[1], fids)
            indexes = np.where(mask)[0]
        elif isinstance(fids, list):
            mask = np.in1d(labels[1], fids)
            indexes = np.where(mask)[0]
    elif n_samples:
        indexes = np.arange(len(embeddings))
        indexes = np.random.choice(indexes, size=n_samples, replace=False)

    X = fused_concat_embeddings[indexes]
    labels = [label[indexes] for label in labels]

    print("Setting up t-SNE...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate="auto", n_iter=5000, verbose=True)
    tsne_results = tsne.fit_transform(X)

    print(str(tsne) + f" with n_components={n_components} and {n_samples} embeddings.")

    # Create markers based on kin_relations
    # labels -> (kin_relations, face1_famliy_id, face2_family_id)
    # where kin_relations are 11 possible strings: bb, ss, sibs, ms, md, fs, fd, gfgs, gfgd, gmgs, gmgd
    marker_colors = np.where(labels[1] == labels[2], "blue", "red").reshape(
        -1,
    )

    if n_components == 2:
        scatter = go.Scatter(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            mode="markers",
            marker=dict(color=marker_colors, size=10),
            text=[f"F1 FID: {face1_id}, F2 FID: {face2_id}, Kinship: {kr}" for kr, face1_id, face2_id in zip(*labels)],
            hoverinfo="text",
        )
    elif n_components == 3:
        scatter = go.Scatter3d(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            z=tsne_results[:, 2],
            mode="markers",
            marker=dict(color=marker_colors, size=3),
            text=[
                f"F1 FID: {face1_id[0]}, F2 FID: {face2_id[0]}, Kinship: {kr[0]}"
                for kr, face1_id, face2_id in zip(*labels)
            ],
            hoverinfo="text",
        )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=go.layout.Margin(
            l=0, r=0, b=0, t=0, pad=4  # left margin  # right margin  # bottom margin  # top margin  # padding
        ),
    )
    fig.show()


# %%
plot_tsne(n_components=3, fids=[250, 283, 409, 735, 873], perplexity=100)

# %%
plot_tsne(n_components=3, fids=10, perplexity=100)
