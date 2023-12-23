"""Train model for kinship verification using SOTA2020 strategies."""
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchmetrics as tm
from datasets.mtcf import MTCFDataset
from models.mtcf import MTCFNet
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import predict_kinship_mtcf, test_pairs, update_lr_mtcf, validate_pairs


def log(loss, metric, epoch, step, global_step, cur_lr):
    accuracy = metric.compute()
    s = (
        f"epoch: {epoch + 1:>2} | step: {global_step + 1:>5} | "
        f"loss: {loss:.3f} | acc: {accuracy:.3f} | lr: {cur_lr:.5f}"
    )
    if step == 0 or step % args.loss_log_step == args.loss_log_step - 1:  # Print every 100 mini-batches
        print(s)


def train(args):
    # Define transformations for training and validation sets
    transform_img_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            # left-right filliping, random contrast, brightness, saturation with a probability of 0.5
            transforms.RandomApply(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                ],
                p=0.5,
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
    train_dataset = MTCFDataset(Path(args.root_dir), Path(args.train_dataset_path), transform=transform_img_train)
    # Define the training dataset
    val_dataset = MTCFDataset(Path(args.root_dir), Path(args.val_dataset_path), transform=transform_img_val)

    # Define the model
    model = MTCFNet(weights=args.weights)
    model.to(device)

    # Define the metric
    metric = tm.Accuracy(task="binary")
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,  # Assuming 12 workers for loading data
        pin_memory=True,
    )

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.comparator.parameters(), lr=args.start_lr, weight_decay=args.l2_factor)
    bce_loss = nn.BCEWithLogitsLoss()

    total_steps = len(train_loader) * args.num_epoch
    print(f"Total steps = {total_steps}")
    best_auc = 0.0

    # Training loop
    for epoch in range(args.num_epoch):
        model.train()
        metric.reset()
        epoch_loss = 0.0

        # Update learning rate
        update_lr_mtcf(optimizer, epoch, args.end_lr)

        for step, (img1, img2, labels) in enumerate(train_loader):
            global_step = step + epoch * len(train_loader)
            # Transfer to GPU if available
            img1, img2 = img1.to(device), img2.to(device)
            kin_1hot, is_kin = labels
            kin_1hot, is_kin = kin_1hot.to(device), is_kin.to(device=device, dtype=torch.float32)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(img1, img2, kin_1hot)
            z = torch.max(predictions, dim=1)[0]

            # Compute metric
            metric(z, is_kin)

            # Compute loss
            loss = bce_loss(z, is_kin)

            # Backward pass and optimize
            loss.backward()

            # Print statistics
            cur_lr = optimizer.param_groups[0]["lr"]
            step_loss = loss.item()
            epoch_loss += step_loss
            log(step_loss, metric, epoch, step, global_step, cur_lr)
            # Update parameters
            optimizer.step()

        # Save model checkpoints
        auc, thresh = validate_pairs(
            model, val_dataloader, device=args.device, return_thresh=True, predict=predict_kinship_mtcf
        )
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), args.output_dir / "best.pth")

        accuracy = metric.compute()
        print(
            f"epoch {epoch + 1:02} | epoch_acc: {accuracy:.3f} "
            + f"| epoch_loss: {epoch_loss / len(train_loader):.3f} | epoch_auc: {auc:.3f} | thresh: {thresh:.3f}"
        )


def val(args):
    transform_img_val = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the training dataset
    val_dataset = MTCFDataset(Path(args.root_dir), Path(args.dataset_path), transform=transform_img_val)

    # Define the model
    model = MTCFNet()
    # Load the model weights
    model.load_state_dict(torch.load(args.weights))
    model.to(device)

    # Define the DataLoader for the training set
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,  # Assuming 12 workers for loading data
        pin_memory=True,
    )

    auc, thresh = validate_pairs(
        model, val_dataloader, device=args.device, return_thresh=True, predict=predict_kinship_mtcf
    )

    print(f"auc: {auc:.3f} | thresh: {thresh:.3f}")


def test(args):
    transform_img_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the training dataset
    test_dataset = MTCFDataset(
        Path(args.root_dir),
        Path(args.dataset_path),
        negatives_per_sample=0,
        extend_with_same_gen=False,
        transform=transform_img_test,
    )

    # Define the model
    model = MTCFNet()
    # Load the model weights
    model.load_state_dict(torch.load(args.weights))
    model.to(device)

    # Define the DataLoader for the training set
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,  # Assuming 12 workers for loading data
        pin_memory=True,
    )

    acc = test_pairs(model, test_dataloader, device=args.device, thresh=args.threshold, predict=predict_kinship_mtcf)

    print(f"acc: {acc:.3f}")


def create_parser_train(subparsers):
    parser = subparsers.add_parser("train", help="Train the model")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--train-dataset-path", type=str, required=True)
    parser.add_argument("--val-dataset-path", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--start-lr", type=float, default=1e-3, help="Start learning rate")
    parser.add_argument("--end-lr", type=float, default=5e-4, help="End learning rate")
    parser.add_argument("--l2-factor", type=float, default=2e-4, help="Weight decay")
    parser.add_argument("--num-epoch", type=int, default=4, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--loss-log-step", type=int, default=100, help="Steps for logging loss")
    parser.add_argument("--device", type=str, default="0", help="Device to use for training")
    parser.set_defaults(func=train)


def create_parser_val(subparsers):
    parser = subparsers.add_parser("val", help="Train the model")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device to use for training")
    parser.set_defaults(func=val)


def create_parser_test(subparsers):
    parser = subparsers.add_parser("test", help="Train the model")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device to use for training")
    parser.set_defaults(func=test)


if __name__ == "__main__":
    # Set random seed to 100
    torch.manual_seed(100)

    parser = ArgumentParser(description="Configuration for the MTCF strategy")
    subparsers = parser.add_subparsers()

    create_parser_train(subparsers)
    create_parser_val(subparsers)
    create_parser_test(subparsers)

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

    args.func(args)
