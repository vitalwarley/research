from argparse import ArgumentParser
from pathlib import Path

import torch
import torchmetrics as tm
from losses import facornet_contrastive_loss
from models.facornet import FaCoR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import TQDM_BAR_FORMAT, set_seed

from datasets.facornet import FIWFaCoRNet as FIW


# Validation loop
def predict(model, val_loader, device=0):
    similarities = []
    # logits_list = []
    y_true = []
    y_true_kin_relations = []

    with torch.no_grad():
        for i, (img1, img2, labels) in tqdm(enumerate(val_loader), total=len(val_loader), bar_format=TQDM_BAR_FORMAT):
            # Transfer to GPU if available
            img1, img2 = img1.to(device), img2.to(device)
            (kin_relation, is_kin) = labels
            labels = (kin_relation.to(device), is_kin.to(device))

            f1, f2, _ = model([img1, img2])
            sim = torch.cosine_similarity(f1, f2).detach()
            similarities.append(sim)
            y_true.append(is_kin)
            y_true_kin_relations.append(kin_relation)

    # Concat
    similarities = torch.concatenate(similarities)
    y_true = torch.concatenate(y_true)
    y_true_kin_relations = torch.concatenate(y_true_kin_relations)

    return similarities, y_true


def validate(model, dataloader, device=0):
    model.eval()
    # Compute similarities
    similarities, y_true = predict(model, dataloader)
    # Move all to device
    similarities = similarities.to(device)
    y_true = y_true.to(device)
    # Compute metrics
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
    acc_metric = tm.Accuracy(task="binary", threshold=threshold).to(device)
    acc = acc_metric(similarities, y_true)
    return auc, threshold, acc


def train(args):
    # Define transformations for training and validation sets
    # Did they mentioned augmentations?
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = FIW(root_dir=args.root_dir, sample_path=Path(FIW.TRAIN_PAIRS), transform=transform)
    val_model_sel_dataset = FIW(root_dir=args.root_dir, sample_path=Path(FIW.VAL_PAIRS_MODEL_SEL), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    val_model_sel_loader = DataLoader(
        val_model_sel_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False
    )

    model = FaCoR()
    model.to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_factor)

    total_steps = len(train_loader)
    print(f"Total steps: {total_steps}")
    global_step = 0
    best_model_auc, _, val_acc = validate(model, val_model_sel_loader)
    print(f"epoch: 0 | auc:  {best_model_auc:.6f} | acc: {val_acc:.6f}")

    for epoch in range(args.num_epoch):
        model.train()
        loss_epoch = 0.0
        for step, data in enumerate(train_loader):
            global_step = step + epoch * args.steps_per_epoch

            image1, image2, labels = data
            (kin_relation, is_kin) = labels

            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
            kin_relation = kin_relation.to(args.device)
            is_kin = is_kin.to(args.device)

            x1, x2, att = model([image1, image2])
            loss = facornet_contrastive_loss(x1, x2, beta=att)

            loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) == args.steps_per_epoch:
                break

        use_sample = (epoch + 1) * args.batch_size * args.steps_per_epoch
        train_dataset.set_bias(use_sample)

        # Save model checkpoints
        auc, _, val_acc = validate(model, val_model_sel_loader)

        if auc > best_model_auc:
            best_model_auc = auc
            torch.save(model.state_dict(), args.output_dir / "best.pth")

        print(
            f"epoch: {epoch + 1:>2} | step: {global_step} "
            + f"| loss: {loss_epoch / args.steps_per_epoch:.3f} | auc: {auc:.6f} | acc: {val_acc:.6f}"
        )


def create_parser_train(subparsers):
    parser = subparsers.add_parser("train", help="Train the model")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-epoch", type=int, default=40, help="Number of epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=50, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--device", type=str, default="0", help="Device to use for training")
    parser.set_defaults(func=train)


if __name__ == "__main__":
    # Necessary for dataloaders?
    torch.multiprocessing.set_start_method("spawn")
    set_seed(42)

    parser = ArgumentParser(description="Configuration for the FaCoRNet strategy")
    subparsers = parser.add_subparsers()
    create_parser_train(subparsers)
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(args)

    # Write args to args.yaml
    with open(args.output_dir / "args.yaml", "w") as f:
        f.write(str(args))

    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.device}")
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current CUDA Device = {current_device}")
        print(f"Device Name = {device_name}")
    else:
        print("CUDA is not available.")

    train(args)
