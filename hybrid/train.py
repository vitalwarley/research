from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torchmetrics as tm
import transforms as mytransforms
from dataset import FamiliesDataset
from model import InsightFace
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms

NUM_CLASSES: int = 570  # FIW train families
EMBEDDING_DIM: int = 512
LR: float = 1e-4
START_LR: float = 1e-10
END_LR: float = 1e-10
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 1e-4
LR_STEPS: tuple = (
    8,
    14,
    25,
    35,
    40,
    50,
    60,
)  # Epochs at which to decrease learning rate, however we have only 20 epochs...
LR_FACTOR: float = 0.75
WARMUP: int = 200
COOLDOWN: int = 400
NUM_EPOCH: int = 20
CLIP_GRADIENT: float = 1.0  # Differ from paper

JITTER_PARAM: float = 0.15
LIGHTING_PARAM: float = 0.15

LOSS_LOG_STEP: int = 100


def update_lr(optimizer, global_step, total_steps):
    if global_step < WARMUP:
        cur_lr = (global_step + 1) * (LR - START_LR) / WARMUP + START_LR
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
    # cool down lr
    elif global_step > total_steps - COOLDOWN:  # cooldown start
        # TODO: why only the first param group? what are the other param groups?
        # TODO: I should experiment with updating all param groups
        # There is only one param group.
        cur_lr = (total_steps - global_step) * (optimizer.param_groups[0]["lr"] - END_LR) / COOLDOWN + END_LR
        optimizer.param_groups[0]["lr"] = cur_lr


def log(loss, metric, epoch, step, global_step, cur_lr, args):
    accuracy = metric.compute()
    s = (
        f"Epoch {epoch + 1:>2} | Step {global_step + 1:>5} - "
        f"Loss: {loss:.3f}, Acc: {accuracy:.3f}, LR: {cur_lr:.10f}"
    )
    if step % LOSS_LOG_STEP == LOSS_LOG_STEP - 1:  # Print every 100 mini-batches
        print(s)
    with open(f"{args.output_dir}/log.txt", "a") as f:
        f.write(s + "\n")


def train(args):
    # Set random seed to 100
    torch.manual_seed(100)

    # Define transformations for training and validation sets
    transform_img_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ColorJitter(
                brightness=JITTER_PARAM,
                contrast=JITTER_PARAM,
                saturation=JITTER_PARAM,
            ),
            mytransforms.Lightning(
                LIGHTING_PARAM,
                mytransforms._IMAGENET_PCA["eigval"],
                mytransforms._IMAGENET_PCA["eigvec"],
            ),
            transforms.ToTensor(),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the training dataset
    train_dataset = FamiliesDataset(Path(args.dataset_path), transform=transform_img_train)
    num_classes = len(train_dataset.families)  # This should be the number of families or classes

    # Define the model
    model = InsightFace(num_classes=num_classes, weights=args.insightface_weights, normalize=args.normalize)
    model.to(device)
    model.train()

    # Define the metric
    metric = tm.Accuracy(task="multiclass", num_classes=num_classes)
    metric.to(device)

    # Define the DataLoader for the training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=48,  # Assuming a batch size of 48 as in the original script
        shuffle=True,
        num_workers=12,  # Assuming 12 workers for loading data
        pin_memory=True,
    )

    # Define the optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=START_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=LR_STEPS, gamma=LR_FACTOR)
    loss_function = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * NUM_EPOCH

    print(f"Total number of steps: {total_steps}")

    epoch_begin_ts = datetime.now()
    print(f"Start training at {epoch_begin_ts.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.normalize:
        model_path = Path(args.output_dir) / "model_epoch_{}_norm.pth"
    else:
        model_path = Path(args.output_dir) / "model_epoch_{}.pth"

    # Training loop
    for epoch in range(NUM_EPOCH):
        epoch_begin_ts = datetime.now()
        metric.reset()
        for step, (img, family_idx, _) in enumerate(train_loader):
            global_step = step + epoch * len(train_loader)
            # Transfer to GPU if available
            inputs, labels = img.to(device), family_idx.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute metric
            metric(outputs, labels)

            # Compute loss
            loss = loss_function(outputs, labels)

            # Backward pass and optimize
            loss.backward()

            if args.normalize:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRADIENT)

            # Print statistics
            cur_lr = optimizer.param_groups[0]["lr"]
            log(loss.item(), metric, epoch, step, global_step, cur_lr, args)
            # Update learning rate (warmup or cooldown only)
            update_lr(optimizer, global_step, total_steps)
            # Update parameters
            optimizer.step()

        scheduler.step()

        # Save model checkpoints
        if epoch % 10 == 9:  # Save every 10 epochs
            torch.save(model.state_dict(), str(model_path).format(epoch + 1))

        # Print epoch accuracy
        epoch_end_ts = datetime.now()
        epoch_total_time = epoch_end_ts - epoch_begin_ts
        accuracy = metric.compute()
        steps_per_second = len(train_loader) / epoch_total_time.total_seconds()
        print(f"Epoch {epoch + 1:02} - Acc: {accuracy:.3f}, Time: {epoch_total_time}, Steps/s: {steps_per_second:.3f}")

    print(f"Finished training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--insightface-weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    # Get total experiments in output_dir
    num_experiments = len(list(args.output_dir.glob("*")))
    # Create output directory
    now = datetime.now()
    args.output_dir = args.output_dir / f"{num_experiments + 1}_{now.strftime('%Y%m%d%H%M%S')}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write args to args.yaml
    with open(args.output_dir / "args.yaml", "w") as f:
        f.write(str(args))

    train(args)
