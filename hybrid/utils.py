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


# Validation loop for verification task
def predict_kinship(model, val_loader, device=0):
    similarities = []
    y_true = []

    with torch.no_grad():
        for _, (img1, img2, labels) in tqdm(enumerate(val_loader), total=len(val_loader), bar_format=TQDM_BAR_FORMAT):
            # Transfer to GPU if available
            img1, img2 = img1.to(device), img2.to(device)
            labels = labels.to(device)

            f1 = model(img1)
            f2 = model(img2)

            sim = F.cosine_similarity(f1, f2).detach()
            similarities.append(sim)
            y_true.append(labels)

    # Concat
    similarities = torch.concatenate(similarities)
    y_true = torch.concatenate(y_true)

    return similarities, y_true


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


def validate_pairs(model, dataloader, device, return_thresh=False):
    model.eval()
    predictions, y_true = predict_kinship(model, dataloader)
    # move all to device
    predictions = predictions.to(device)
    y_true = y_true.to(device)
    # compute metrics
    auroc = tm.AUROC(task="binary").to(device)
    auc = auroc(predictions, y_true)
    if return_thresh:
        fpr, tpr, thresholds = tm.functional.roc(predictions, y_true, task="binary")
        diff: torch.Tensor = tpr - fpr
        maxindex = diff.argmax()
        threshold = thresholds[maxindex]
        if threshold.isnan().item():
            threshold = 0.01
        else:
            threshold = threshold.item()
        return auc, threshold
    return auc


def test(model, dataloader, threshold):
    model.eval()
    predictions, y_true = predict(model, dataloader)
    acc = tm.functional.accuracy(predictions, y_true, threshold=threshold, task="binary")
    return acc


def update_lr(optimizer, global_step, total_steps, args):
    if global_step < args.warmup:
        cur_lr = (global_step + 1) * (args.lr - args.start_lr) / args.warmup + args.start_lr
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
    # cool down lr
    elif global_step > total_steps - args.cooldown:  # cooldown start
        # TODO: why only the first param group? what are the other param groups?
        # TODO: args.i should experiment with updating all param groups
        # There is only one param group.
        cur_lr = (total_steps - global_step) * (
            optimizer.param_groups[0]["lr"] - args.end_lr
        ) / args.cooldown + args.end_lr
        optimizer.param_groups[0]["lr"] = cur_lr


def contrastive_loss(x1, x2, beta=0.08):
    x1x2 = torch.cat([x1, x2], dim=0)
    x2x1 = torch.cat([x2, x1], dim=0)

    cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1), torch.unsqueeze(x1x2, dim=0), dim=2) / beta
    mask = 1.0 - torch.eye(2 * x1.size(0)).to(x1.device)
    numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / beta)
    denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
    return -torch.mean(torch.log(numerators / denominators), dim=0)


def supervised_contrastive_loss(embeddings, labels, tau=1.0, eps=1e-8):
    """
    Supervised Contrastive Learning Loss function.

    Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning (Gunel et al., 2021)

    Correspondence to the formula components:
    - exp_logits corresponds to the denominator of the contrastive loss formula.
    - mask_positive corresponds to the indicator function 1_{i!=j}1_{y_i=y_j}.
    - log_prob corresponds to the log(exp(...)) part of the formula.
    - mean_log_prob_pos corresponds to the average log probability over positive pairs.

    Args:
    - embeddings (torch.Tensor): The embeddings matrix (batch_size x embedding_size).
    - labels (torch.Tensor): The labels vector (batch_size).
    - tau (float): The temperature parameter.
    - eps (float): A small epsilon value to prevent division by zero in log.

    Returns:
    - torch.Tensor: The calculated loss.
    """

    # Normalize the embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute the similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / tau

    # Create the mask for positive and negative examples
    labels = labels.unsqueeze(0)
    mask_positive = torch.eq(labels, labels.T).float()  # 1_{y_i=y_j}
    mask_negative = 1 - mask_positive  # Ensures i != j

    # Mask out the diagonal part of the matrix since we don't use i=j pairs
    mask_positive.fill_diagonal_(0)

    # Calculate the log probabilities
    exp_logits = torch.exp(similarity_matrix) * mask_negative  # Sum part of the denominator
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + eps)  # Logarithm of exp

    # Calculate the mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask_positive * log_prob).sum(1) / (mask_positive.sum(1) + eps)  # Numerator of the loss

    # Calculate the loss
    loss = -mean_log_prob_pos.mean()  # Final SCL loss

    return loss
