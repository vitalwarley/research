"""Train model for kinship verification using SOTA2020 strategies."""

from argparse import ArgumentParser
from pathlib import Path

import torch
import torchmetrics as tm
import transforms as mytransforms
from dataset import FamiliesDataset, PairDataset
from model import InsightFace
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import supervised_contrastive_loss, update_lr, validate_pairs


def log(loss, metric, epoch, step, global_step, cur_lr, output_dir):
    accuracy = metric.compute()
    s = (
        f"epoch: {epoch + 1:>2} | step: {global_step + 1:>5} | "
        f"loss: {loss:.3f} | acc: {accuracy:.3f} | lr: {cur_lr:.10f}"
    )
    if step == 0 or step % args.loss_log_step == args.loss_log_step - 1:  # Print every 100 mini-batches
        print(s)
    with open(f"{output_dir}/log.txt", "a") as f:
        f.write(s + "\n")


def train(args):
    # Set random seed to 100
    torch.manual_seed(100)

    # Define transformations for training and validation sets
    transform_img_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ColorJitter(
                brightness=args.jitter_param,
                contrast=args.jitter_param,
                saturation=args.jitter_param,
            ),
            mytransforms.Lightning(
                args.lighting_param,
                mytransforms._IMAGENET_PCA["eigval"],
                mytransforms._IMAGENET_PCA["eigvec"],
            ),
            transforms.ToTensor(),
        ]
    )
    transform_img_val = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the training dataset
    train_dataset = FamiliesDataset(Path(args.train_dataset_path), transform=transform_img_train)
    num_classes = len(train_dataset.families)  # This should be the number of families or classes
    fam_dataset = FamiliesDataset(Path(args.val_dataset_path), transform=transform_img_val)
    val_dataset = PairDataset(fam_dataset)  # Reads val_pairs.csv

    # Define the model
    model = InsightFace(num_classes=num_classes, weights=args.insightface_weights, normalize=args.normalize)
    model.to(device)

    # Define the metric
    metric = tm.Accuracy(task="multiclass", num_classes=num_classes)
    metric.to(device)

    # Define the DataLoader for the training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,  # Assuming 12 workers for loading data
        pin_memory=True,
    )

    # Define the DataLoader for the training set
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // 2,
        shuffle=False,
        num_workers=12,  # Assuming 12 workers for loading data
        pin_memory=True,
    )

    # Define the optimizer and loss function
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.start_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_factor)
    ce_loss = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * args.num_epoch
    best_auc = 0.0

    # Training loop
    for epoch in range(args.num_epoch):
        model.train()
        metric.reset()
        epoch_loss = 0.0
        for step, (img, family_idx, _) in enumerate(train_loader):
            global_step = step + epoch * len(train_loader)
            # Transfer to GPU if available
            inputs, labels = img.to(device), family_idx.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            features, outputs = model(inputs, return_features=True)

            # Compute metric
            metric(outputs, labels)

            # Compute loss
            if args.scl:
                loss = (1 - args.scl_lambda) * supervised_contrastive_loss(
                    features, labels, tau=args.tau
                ) + args.scl_lambda * ce_loss(outputs, labels)
            else:
                loss = ce_loss(outputs, labels)

            # Backward pass and optimize
            loss.backward()

            if args.clip_gradient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)

            # Print statistics
            cur_lr = optimizer.param_groups[0]["lr"]
            step_loss = loss.item()
            epoch_loss += step_loss
            log(step_loss, metric, epoch, step, global_step, cur_lr, args.output_dir)
            # Update learning rate (warmup or cooldown only)
            update_lr(optimizer, global_step, total_steps, args)
            # Update parameters
            optimizer.step()

        scheduler.step()

        # Save model checkpoints
        auc = validate_pairs(model, val_dataloader, device=args.device)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), args.output_dir / "best.pth")

        accuracy = metric.compute()
        print(
            f"epoch {epoch + 1:02} | epoch_acc: {accuracy:.3f} "
            + f"| epoch_loss: {epoch_loss / len(train_loader):.3f} | epoch_auc: {auc:.3f}"
        )


def create_parser():
    parser = ArgumentParser(description="Configuration for the training script")

    parser.add_argument("--train-dataset-path", type=str, required=True)
    parser.add_argument("--val-dataset-path", type=str, required=True)
    parser.add_argument("--insightface-weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--num-classes", type=int, default=570, help="Number of classes")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Dimension of the embedding")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--start-lr", type=float, default=1e-10, help="Start learning rate")
    parser.add_argument("--end-lr", type=float, default=1e-10, help="End learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--lr-steps", type=int, nargs="+", default=[8, 14, 25, 35, 40, 50, 60], help="Epochs to decrease learning rate"
    )
    parser.add_argument("--lr-factor", type=float, default=0.75, help="Learning rate decrease factor")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations")
    parser.add_argument("--cooldown", type=int, default=400, help="Cooldown iterations")
    parser.add_argument("--num-epoch", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--clip-gradient", type=float, default=1.5, help="Gradient clipping")
    parser.add_argument("--jitter-param", type=float, default=0.15, help="Jitter parameter")
    parser.add_argument("--lighting-param", type=float, default=0.15, help="Lighting parameter")
    parser.add_argument("--loss-log-step", type=int, default=100, help="Steps for logging loss")
    parser.add_argument("--tau", type=float, default=0.3, help="Temperature parameter for supervised contrastive loss")
    parser.add_argument("--scl", action="store_true", help="Use SCL")
    parser.add_argument("--scl-lambda", type=float, default=0.9, help="Lambda for SCL")
    parser.add_argument("--device", type=str, default="0", help="Device to use for training")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(args)

    # Write args to args.yaml
    with open(args.output_dir / "args.yaml", "w") as f:
        f.write(str(args))

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current CUDA Device = {current_device}")
        print(f"Device Name = {device_name}")
        args.device = torch.device(f"cuda:{args.device}")
    else:
        print("CUDA is not available.")
        args.device = torch.device("cpu")

    train(args)
