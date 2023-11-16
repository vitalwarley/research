from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dataset import FamiliesDataset, PairDataset
from matplotlib import pyplot as plt
from model import FamilyClassifier, InsightFace
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


# Validation loop
def predict(model, val_loader, device=0):
    similarities = []
    y_true = []
    for i, (img1, img2, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
        # Transfer to GPU if available
        img1, img2 = img1.to(device), img2.to(device)

        # Forward pass
        (emb1, _) = model(img1, return_features=True)
        (emb2, _) = model(img2, return_features=True)

        sim = F.cosine_similarity(emb1, emb2).detach().cpu().numpy()
        similarities.append(sim)
        y_true.append(label)

    # Concat
    similarities = np.concatenate(similarities)
    y_true = np.concatenate(y_true)

    return similarities, y_true


def val(args):
    transform_img_val = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the validation dataset
    fam_dataset = FamiliesDataset(args.dataset_path, transform=transform_img_val)
    dataset = PairDataset(fam_dataset)  # Reads val_pairs.csv

    # Define the DataLoader for the training set
    dataloader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=False,
        num_workers=12,  # Assuming 12 workers for loading data
        pin_memory=True,
    )

    print(f"Validating with {len(dataset)} pairs.")

    num_classes = 570  # num training classes

    # Baseline
    model = InsightFace(num_classes=num_classes, weights=args.baseline_weights)
    model.to(device)
    model.eval()
    predictions, y_true = predict(model, dataloader)
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(fpr, tpr, color="b", lw=2, label=f"baseline AUC:{auc(fpr, tpr):.4f}")
    print(f"Baseline AUC: {auc(fpr, tpr):.4f}")

    # + Classification
    model = FamilyClassifier(num_classes=num_classes, weights=args.classification_weights)
    model.to(device)
    model.eval()
    predictions, y_true = predict(model, dataloader)
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(fpr, tpr, color="g", lw=2, label=f" +classification AUC:{auc(fpr, tpr):.4f}")
    print(f"Classification AUC: {auc(fpr, tpr):.4f}")

    # + Normalization
    model = FamilyClassifier(num_classes=num_classes, weights=args.normalization_weights, normalize=True)
    model.to(device)
    model.eval()
    predictions, y_true = predict(model, dataloader)
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(fpr, tpr, color="r", lw=2, label=f" +normalization AUC:{auc(fpr, tpr):.4f}")
    print(f"Normalization AUC: {auc(fpr, tpr):.4f}")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"{args.logdir}/roc.png", pad_inches=0, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--baseline-weights", type=str, required=True)
    parser.add_argument("--classification-weights", type=str, required=True)
    parser.add_argument("--normalization-weights", type=str, required=True)
    parser.add_argument("--logdir", type=str, required=True)

    args = parser.parse_args()

    args.logdir = Path(args.logdir)
    # Get total experiments in logdir
    num_experiments = len(list(args.logdir.glob("*")))
    # Create output directory
    now = datetime.now()
    args.logdir = args.logdir / f"{num_experiments + 1}_{now.strftime('%Y%m%d%H%M%S')}"
    args.logdir.mkdir(parents=True, exist_ok=True)

    args.dataset_path = Path(args.dataset_path)

    # Write args to args.yaml
    with open(args.logdir / "args.yaml", "w") as f:
        f.write(str(args))

    val(args)
