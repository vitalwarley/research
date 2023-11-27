import torch
import torch.nn.functional as F
import torchmetrics as tm
from tqdm import tqdm

TQDM_BAR_FORMAT = "Validating... {bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


# Validation loop
def predict(model, val_loader, device=0):
    similarities = []
    y_true = []

    for i, (img1, img2, label) in tqdm(enumerate(val_loader), total=len(val_loader), bar_format=TQDM_BAR_FORMAT):
        # Transfer to GPU if available
        img1, img2 = img1.to(device), img2.to(device)
        label = label.to(device)

        (emb1, _) = model(img1, return_features=True)
        (emb2, _) = model(img2, return_features=True)

        sim = F.cosine_similarity(emb1, emb2).detach()
        similarities.append(sim)
        y_true.append(label)

    # Concat
    similarities = torch.concatenate(similarities)
    y_true = torch.concatenate(y_true)

    return similarities, y_true


def validate(model, dataloader):
    model.eval()
    predictions, y_true = predict(model, dataloader)
    auroc = tm.AUROC(task="binary")
    auc = auroc(predictions, y_true)
    fpr, tpr, thresholds = tm.functional.roc(predictions, y_true, task="binary")
    diff: torch.Tensor = tpr - fpr
    maxindex = diff.argmax()
    threshold = thresholds[maxindex]
    if threshold.isnan().item():
        threshold = 0.01
    else:
        threshold = threshold.item()
    return auc, threshold


def test(model, dataloader, threshold):
    model.eval()
    predictions, y_true = predict(model, dataloader)
    acc = tm.functional.accuracy(predictions, y_true, threshold=threshold, task="binary")
    return acc
