import matplotlib.pyplot as plt
import torch
from lightning import seed_everything
from models.facornet import FaCoRNetLightning
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.facornet import FaCoRNetDataModule

seed_everything(100)

num_epochs = 10
batch_size = 20
steps_per_epoch = 50

# Step 1: Initialization
model = FaCoRNetLightning()
model.cuda()

# Step 2: Data Preparation
data_module = FaCoRNetDataModule(batch_size=batch_size, root_dir="../datasets/facornet")
data_module.setup("fit")
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()


# Step 3: Training Loop
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for idx, batch in enumerate(tqdm(train_loader, total=steps_per_epoch)):
        optimizer.zero_grad()
        loss = model.training_step(batch, idx)
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
        for batch in tqdm(val_loader):
            val_loss = model.validation_step(batch, ...)
            # Log validation loss
            epoch_val_loss += val_loss.item()
        epoch_val_loss /= len(val_loader)

    epoch_train_loss /= len(train_loader)
    print(f"Epoch: {epoch} | Loss: {epoch_train_loss} | Val Loss: {epoch_val_loss}")

    use_sample = (epoch + 1) * batch_size * steps_per_epoch
    train_loader.dataset.set_bias(use_sample)
