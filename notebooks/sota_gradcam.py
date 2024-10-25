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

import matplotlib.patches as patches

# %%
import matplotlib.pyplot as plt
import numpy as np
import pacmap
import torch
import torch.nn.functional as F
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
# %matplotlib inline

# %%
# %load_ext autoreload
# %autoreload 2


torch.manual_seed(100)

# %%
import scienceplots

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
from dataset import FIW, FIWPair  # noqa: E402
from ours.models.base import SimpleModel  # noqa: E402
from ours.models.rfiw2021 import RFIW2021Net  # noqa: E402
from ours.models.scl import SCL, SCLRFIW2021  # noqa: E402


# %%
def setup(root_dir, ckpt_path, csv_path, batch_size, samples_per_member, gpu: int = 0):
    # Loading and sampling the dataset
    val_dataset = FIWPair(
        root_dir=root_dir, csv_path=csv_path, families=[22, 40, 44, 53, 63], samples_per_member=samples_per_member
    )
    # val_dataset = FIW(root_dir=root_dir, families=[], samples_per_member=samples_per_member)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=False)

    # Loading model
    checkpoint = torch.load(ckpt_path)
    simple_model = SimpleModel("adaface_ir_101", projection=None)
    model = SCL(model=simple_model, loss=None)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.cuda()

    model = model.model.backbone

    return model, val_loader


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
# %matplotlib inline
if IS_NOTEBOOK:
    if __name__ == "__main__":
        args = parser()
        run(**vars(args))
else:
    root_dir = Path(HERE, "../datasets/facornet/images/")
    csv_path = Path(HERE, "../datasets/facornet/txt/val_choose_A.txt")
    batch_size = 40
    experiment = "85a4d335a0f5427eaa00539397bdfcb0"
    checkpoint = "7-2.971-1.659-0.876753.ckpt"
    ckpt_path = Path(Path.home(), f".guild/runs/{experiment}/exp/checkpoints/{checkpoint}")
    # ckpt_path = Path(HERE, f"../ours/weights/model_track1.pth")
    plot_path = "plots_experiments/sota_gradcam.png"
    model, val_loader = setup(root_dir, ckpt_path, csv_path, batch_size, samples_per_member=100)


# %%
class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.hook_backward = module.register_full_backward_hook(self.hook_fn_backward)
        self.features = []
        self.gradients = []

    def hook_fn(self, module, input, output):
        self.features.append(output)

    def hook_fn_backward(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def remove(self):
        self.hook.remove()
        self.hook_backward.remove()


def grad_cam(model, face1, face2, target_layer, device):
    model.zero_grad()
    face1 = face1.unsqueeze(0).to(device) / 255.0
    face2 = face2.unsqueeze(0).to(device) / 255.0

    # Register hooks
    activated_features = SaveFeatures(target_layer)

    # Get model output (features and feature_map) for both faces
    features1, _ = model(face1)
    features2, _ = model(face2)

    # Normalize
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)

    # Compute similarity score (you might need to adjust this based on your model's specific comparison method)
    similarity_score = F.cosine_similarity(features1, features2)

    # Backward pass
    similarity_score.backward()

    # Process gradients and activations for both faces
    heatmaps = []
    for i in range(2):  # For each face
        gradients = activated_features.gradients[i]
        activations = activated_features.features[i]

        # Pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients
        for j in range(activations.shape[1]):
            activations[:, j, :, :] *= pooled_gradients[j]

        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # ReLU on top of the heatmap
        heatmap = F.relu(heatmap)

        # Normalize the heatmap
        heatmap /= torch.max(heatmap)

        # Upsample the heatmap to the size of the input image
        heatmap = (
            F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=face1.shape[2:], mode="bilinear", align_corners=False)
            .squeeze()
            .detach()
        )

        # Create a 3-channel heatmap
        heatmap_rgb = plt.cm.jet(heatmap.cpu().numpy())[:, :, :3]
        heatmap_rgb = torch.from_numpy(heatmap_rgb).float().permute(2, 0, 1)

        heatmaps.append(heatmap_rgb)

    # Remove hooks
    activated_features.remove()

    return heatmaps, similarity_score


# %%
def torch2numpy(img):
    """Convert a PyTorch tensor image to a NumPy array."""
    img = img.permute(1, 2, 0)
    img = img.clamp(0, 255)
    img = img.to(torch.uint8)
    return img.cpu().numpy()


# %%
import numpy as np


def get_random_pairs(dataset, n=5, kin=False):
    total_items = len(dataset)
    pairs = []

    # Create an array of indexes
    indexes = np.arange(total_items)

    while len(pairs) < n:
        np.random.shuffle(indexes)  # Shuffle the indexes

        # Check each shuffled index
        for idx in indexes:
            item = dataset[idx]
            face1, face2, kin_relation, face1_fid, face2_fid, is_kin = item

            # Add item if it matches the kin condition
            if (kin and is_kin) or (not kin and not is_kin):
                pairs.append(item)

            # Break if we have collected enough pairs
            if len(pairs) == n:
                break

        # If we exhausted all indexes and still don't have enough pairs
        if len(pairs) < n and len(indexes) == total_items:
            print(f"Warning: Only {len(pairs)} valid pairs found, fewer than requested {n}.")
            break

    return pairs[:n]  # Return up to n pairs


# %%
def plot_grad_cam_pairs(pairs, model, target_layer, device, plot_path):
    n = len(pairs)
    fig, axes = plt.subplots(n, 4, figsize=(6 * n, 18))

    for i, (face1, face2, kin_relation, face1_fid, face2_fid, is_kin) in enumerate(pairs):
        print(f"Getting heatmaps for {i+1}/{len(pairs)}: {kin_relation}, {face1_fid}, {face2_fid}, {is_kin}")
        heatmaps, similarity = grad_cam(model, face1, face2, target_layer, device)

        # Compute the correlation between heatmaps
        correlation = np.corrcoef(heatmaps[0].flatten(), heatmaps[1].flatten())[0, 1]

        print(f"Pair Analysis:")
        print(f"  Kin: {is_kin}")
        print(f"  Relation: {kin_relation}")
        print(f"  Family IDs: {face1_fid}, {face2_fid}")
        print(f"  Heatmap Correlation: {correlation:.4f}")
        print()

        face1 = torch2numpy(face1)
        face2 = torch2numpy(face2)

        heatmap1 = heatmaps[0].permute(1, 2, 0).cpu().numpy()
        heatmap2 = heatmaps[1].permute(1, 2, 0).cpu().numpy()

        # Plot face1 original image
        axes[i, 0].imshow(face1)
        axes[i, 0].set_title(f"Face 1 (FID: {face1_fid})", fontsize=18)
        axes[i, 0].axis("off")

        # Plot face2 original image
        axes[i, 1].imshow(face2)
        axes[i, 1].set_title(
            f"Face 2 (FID: {face2_fid}) | Kin: {is_kin}, Relation: {kin_relation}\nSimilarity: {similarity.detach().item():.4f}",
            fontsize=18,
        )
        axes[i, 1].axis("off")

        # Plot face1 with heatmap overlay
        axes[i, 2].imshow(face1)
        axes[i, 2].imshow(heatmap1, alpha=0.5)  # Overlay heatmap
        axes[i, 2].set_title(f"Face 1 Heatmap", fontsize=18)
        axes[i, 2].axis("off")

        # Plot face2 with heatmap overlay
        axes[i, 3].imshow(face2)
        axes[i, 3].imshow(heatmap2, alpha=0.5)  # Overlay heatmap
        axes[i, 3].set_title(f"Face 2 Heatmap", fontsize=18)
        axes[i, 3].axis("off")

    plt.suptitle(f"Kin = {is_kin}", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top to make space for the title
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
    plt.show()
    plt.show()


# %%
# Get the target layer (last layer of the body sequential module)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
target_layer = model.body[-1]

# Get random non-kin pairs
plot_path = "plots_experiments/sota_gradcam_nonkin.png"
non_kin_pairs = get_random_pairs(val_loader.dataset, n=4, kin=False)
plot_grad_cam_pairs(non_kin_pairs, model, target_layer, device, plot_path)

# Get random kin pairs
plot_path = "plots_experiments/sota_gradcam_kin.png"
kin_pairs = get_random_pairs(val_loader.dataset, n=4, kin=True)
plot_grad_cam_pairs(kin_pairs, model, target_layer, device, plot_path)

# %%
