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


def val():
    transform_img_val = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the validation dataset
    HERE = Path(__file__).parent.resolve()
    fam_dataset = FamiliesDataset(Path(HERE, "..", "fitw2020", "val-faces-det"), transform=transform_img_val)
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
    model = InsightFace(num_classes=num_classes, weights="models/ms1mv3_arcface_r100_fp16.pth")
    model.to(device)
    model.eval()
    predictions, y_true = predict(model, dataloader)
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(fpr, tpr, color="b", lw=2, label=f"baseline AUC:{auc(fpr, tpr):.4f}")
    print(f"Baseline AUC: {auc(fpr, tpr):.4f}")

    # + Classification
    model = FamilyClassifier(num_classes=num_classes, weights="arcface_fiw_clf/model_epoch_20.pth")
    model.to(device)
    model.eval()
    predictions, y_true = predict(model, dataloader)
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(fpr, tpr, color="g", lw=2, label=f" +classification AUC:{auc(fpr, tpr):.4f}")
    print(f"Classification AUC: {auc(fpr, tpr):.4f}")

    # + Normalization
    model = FamilyClassifier(num_classes=num_classes, weights="arcface_fiw_clf_norm/model_epoch_20.pth", normalize=True)
    model.to(device)
    model.eval()
    predictions, y_true = predict(model, dataloader)
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.plot(fpr, tpr, color="r", lw=2, label=f" +normalization AUC:{auc(fpr, tpr):.4f}")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("roc.png", pad_inches=0, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    val()
