"""
Adapted from Track1/find.py
"""
import argparse
import glob
import os
import sys

import torch
from torchvision import transforms
from tqdm import tqdm
from Track1.dataset import *
from Track1.losses import *
from Track1.models import *
from Track1.utils import *


class FIW(Dataset):
    def __init__(
        self,
        root_dir: str = "",
        transform: transforms.Compose | None = None,
        families: list | None = None,
        member_limit: int = 10,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.families = [str(i) for i in families]
        self.member_limit = member_limit
        self.sample_list = self.load_sample()

    def load_sample(self):
        sample_list = []
        for family_id in self.families:
            family_path = os.path.join(self.root_dir, f"F0{family_id}")  # FIXME: F0{family_id} is a hack
            member_ids = os.listdir(family_path)[: self.member_limit]
            for member_id in member_ids:
                member_path = os.path.join(family_path, member_id)
                member_images = glob.glob(f"{member_path}/*.jpg")
                for image in member_images:
                    sample_list.append((image, family_id))
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        img = image.load_img(path, target_size=(112, 112))
        return img

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        sample, family_id = self.sample_list[item]
        img = self.read_image(sample)
        if self.transform is not None:
            img = self.transform(img)
        img = np2tensor(self.preprocess(np.array(img, dtype=float)))
        return img, family_id


def extract_embeddings(val_loader, model):
    embeddings = []
    labels = []

    for img, family_id in tqdm(val_loader):
        with torch.no_grad():
            embeddings.append(model.encoder(img.cuda()).cpu().numpy())
            labels.append(family_id)

    # Now, embeddings contain all the embeddings from your model
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)

    return embeddings, labels


def plot_embeddings(embeddings, labels):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)

    # Create a color map for families
    color_map = {250: "red", 283: "blue", 409: "green", 735: "purple", 873: "orange"}

    # Plot
    plt.figure(figsize=(10, 10))
    for label, color in color_map.items():
        idxs = [idx for idx, val in enumerate(labels) if int(val) == label]
        plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], color=color, label=f"Family #{label}")
    # Add axis labels components
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.legend()
    plt.savefig("embeddings.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot embeddings")
    parser.add_argument("--root_dir", type=str, help="root directory of dataset")
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--batch_size", type=int, default=40, help="batch size default 40")
    parser.add_argument("--gpu", default="1", type=str, help="gpu id you use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Loading and sampling the dataset
    val_dataset = FIW(root_dir=args.root_dir, families=[250, 283, 409, 735, 873])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=False)

    # Loading model
    model = Net().cuda()
    model.load_state_dict(torch.load(args.save_path))
    model.eval()

    set_seed(100)

    # Extracting embeddings
    embeddings, labels = extract_embeddings(val_loader, model)

    # Plotting
    plot_embeddings(embeddings, labels)
