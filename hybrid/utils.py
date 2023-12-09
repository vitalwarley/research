import torch
import torch.nn.functional as F
import torchmetrics as tm
from tqdm import tqdm

TQDM_BAR_FORMAT = "Validating... {bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


# Validation loop
def predict(model, val_loader, device=0):
    similarities = []
    logits_list = []
    y_true = []
    y_true_kin_relations = []

    with torch.no_grad():
        for i, (img1, img2, labels) in tqdm(enumerate(val_loader), total=len(val_loader), bar_format=TQDM_BAR_FORMAT):
            # Transfer to GPU if available
            img1, img2 = img1.to(device), img2.to(device)
            (kin_relation, is_kin) = labels
            labels = (kin_relation.to(device), is_kin.to(device))

            f1, p1 = model(img1)
            f2, p2 = model(img2)
            logits = model.classify(p1, p2)

            sim = F.cosine_similarity(f1, f2).detach()
            similarities.append(sim)
            logits_list.append(logits)
            y_true.append(is_kin)
            y_true_kin_relations.append(kin_relation)

    # Concat
    similarities = torch.concatenate(similarities)
    y_true = torch.concatenate(y_true)
    y_true_kin_relations = torch.concatenate(y_true_kin_relations)
    logits = torch.concatenate(logits_list)

    return (similarities, y_true), (logits, y_true_kin_relations)


def validate(model, dataloader, num_classes, device=0):
    model.eval()
    (predictions, y_true), (logits, y_true_kin_relations) = predict(model, dataloader)
    # move all to device
    predictions = predictions.to(device)
    y_true = y_true.to(device)
    y_true_kin_relations = y_true_kin_relations.to(device)
    logits = logits.to(device)
    # compute metrics
    auroc = tm.AUROC(task="binary").to(device)
    acc_metric = tm.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    auc = auroc(predictions, y_true)
    fpr, tpr, thresholds = tm.functional.roc(predictions, y_true, task="binary")
    diff: torch.Tensor = tpr - fpr
    maxindex = diff.argmax()
    threshold = thresholds[maxindex]
    if threshold.isnan().item():
        threshold = 0.01
    else:
        threshold = threshold.item()
    acc = acc_metric(logits, y_true_kin_relations)
    return auc, threshold, acc


def test(model, dataloader, threshold):
    model.eval()
    predictions, y_true = predict(model, dataloader)
    acc = tm.functional.accuracy(predictions, y_true, threshold=threshold, task="binary")
    return acc
