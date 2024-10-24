from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchmetrics as tm
from lightning import seed_everything
from losses import facornet_contrastive_loss
from models.facornet import FaCoR
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from datasets.facornet import FIWFaCoRNet

seed_everything(100)

num_epochs = 10
batch_size = 20
steps_per_epoch = 50

# Step 1: Initialization
model = FaCoR()
model.cuda()

# Step 2: Data Preparation
train_dataset = FIWFaCoRNet(
    root_dir="../datasets/facornet",
    sample_path=Path(FIWFaCoRNet.TRAIN_PAIRS),
    batch_size=batch_size,
    biased=True,
    transform=T.Compose([T.ToTensor()]),
)
val_dataset = FIWFaCoRNet(
    root_dir="../datasets/facornet",
    sample_path=Path(FIWFaCoRNet.VAL_PAIRS_MODEL_SEL),
    batch_size=batch_size,
    transform=T.Compose([T.ToTensor()]),
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# Step 3: Training Loop
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for idx, batch in enumerate(tqdm(train_loader, total=steps_per_epoch)):
        images = batch[:2]
        images = [img.cuda() for img in images]
        outputs = model(images)
        optimizer.zero_grad()
        loss = facornet_contrastive_loss(*outputs)  # f1, f2, att
        loss.backward()
        optimizer.step()
        # Log training loss
        epoch_train_loss += loss.item()
        if idx >= steps_per_epoch:
            break

    # Step 4: Validation Loop
    model.eval()
    with torch.no_grad():
        epoch_val_loss = 0
        dataset_size = len(val_loader.dataset)
        # Preallocate tensors based on the total dataset size
        similarities = torch.zeros(dataset_size, device="cuda")
        y_true = torch.zeros(dataset_size, dtype=torch.uint8, device="cuda")
        for idx, batch in enumerate(tqdm(val_loader)):
            current_batch_size = batch[0].shape[0]
            batch = [b.cuda() for b in batch]
            outputs = model([batch[0], batch[1]])
            # Log validation loss
            val_loss = facornet_contrastive_loss(*outputs)
            epoch_val_loss += val_loss.item()
            # Compute similarities
            f1, f2, att = outputs
            sim = torch.cosine_similarity(f1, f2)
            # Store similarities and labels
            start = current_batch_size * idx
            end = start + current_batch_size
            similarities[start:end] = sim
            y_true[start:end] = batch[-1]

    epoch_train_loss /= steps_per_epoch
    epoch_val_loss /= len(val_loader)

    mid = len(similarities) // 2
    print(f"Similarities: (first={similarities[0]}, mid={similarities[mid]}, last={similarities[-1]})")
    print(f"Kinship: (first={y_true[0]}, mid={y_true[mid]}, last={y_true[-1]})")

    auc = tm.functional.auroc(similarities, y_true, task="binary")
    fpr, tpr, thresholds = tm.functional.roc(similarities, y_true, task="binary")
    # Get the best threshold
    maxindex = (tpr - fpr).argmax()
    threshold = thresholds[maxindex]
    if threshold.isnan().item():
        threshold = 0.01
    else:
        threshold = threshold.item()
    # Compute acc
    acc = tm.functional.accuracy(similarities, y_true, task="binary", threshold=threshold)

    print(
        f"Epoch: {epoch} | Loss: {epoch_train_loss} | Val Loss: {epoch_val_loss} | AUC: {auc} | Acc: {acc} | Threshold: {threshold}"
    )

    use_sample = (epoch + 1) * batch_size * steps_per_epoch
    train_loader.dataset.set_bias(use_sample)
