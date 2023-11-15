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
SCHEDULER: str = "multistep"
LR_STEPS: tuple = (8, 14, 25, 35, 40, 50, 60)
LR_FACTOR: float = 0.75
WARMUP: int = 200
COOLDOWN: int = 400
NUM_EPOCH: int = 20

JITTER_PARAM: float = 0.15
LIGHTING_PARAM: float = 0.15


def train(args):
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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Total number of steps: {total_steps}")

    # Training loop
    for epoch in range(NUM_EPOCH):
        epoch_begin_ts = datetime.now()
        print(f"Start training at {epoch_begin_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        metric.reset()
        running_loss = 0.0
        for step, (img, family_idx, person_idx) in enumerate(train_loader):
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

            if step < WARMUP:
                cur_lr = (step + 1) * (LR - START_LR) / WARMUP + START_LR
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
            # cool down lr
            elif step > total_steps - COOLDOWN:  # cooldown start
                # TODO: why only the first param group? what are the other param groups?
                # TODO: I should experiment with updating all param groups
                cur_lr = (total_steps - step) * (optimizer.param_groups[0]["lr"] - END_LR) / COOLDOWN + END_LR
                optimizer.param_groups[0]["lr"] = cur_lr
            else:
                cur_lr = optimizer.param_groups[0]["lr"]

            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if step % 100 == 99:  # Print every 100 mini-batches
                # s = "[Epoch: %d, Step: %5d] loss: %.3f" % (epoch + 1, step + 1, running_loss / 100)
                # Do the same as the previous line, but in f-string. Also, add accuracy metric.
                s = f"[Epoch: {epoch + 1:>2}, Step: {step + 1:>5}] loss: {running_loss / 100:.3f}, accuracy: {metric.compute():.3f}, lr: {cur_lr:.3e}"
                # Add to log file
                print(s)
                with open(f"{args.output_dir}/log.txt", "a") as f:
                    f.write(s + "\n")
                running_loss = 0.0

        scheduler.step()

        # Save model checkpoints
        if epoch % 10 == 9:  # Save every 10 epochs
            if args.normalize:
                model_path = Path(args.output_dir) / f"model_epoch_{epoch + 1}_norm.pth"
            else:
                model_path = Path(args.output_dir) / f"model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)

        # Print epoch accuracy
        epoch_end_ts = datetime.now()
        epoch_total_time = epoch_end_ts - epoch_begin_ts
        print(
            f"[Epoch {epoch + 1:>2}] accuracy: {metric.compute():.3f}, time: {epoch_total_time} [steps/second: {len(train_loader) / epoch_total_time.total_seconds():.3f}]"
        )

    print(f"Finished training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--insightface-weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    now = datetime.now()
    args.output_dir = Path(args.output_dir) / now.strftime("%Y%m%d%H%M%S")

    train(args)
