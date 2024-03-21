from argparse import ArgumentParser
from pathlib import Path

import torch
import torchmetrics as tm
from datasets.utils import Sample
from losses import facornet_contrastive_loss
from models.facornet import FaCoR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import TQDM_BAR_FORMAT, set_seed

from datasets.facornet import FIWFaCoRNet as FIW


def acc_kr_to_str(out, acc_kr):
    # Add acc_kr to out
    id2name = {v: k for k, v in Sample.NAME2LABEL.items()}
    for kin_id, acc in acc_kr.items():
        kr = id2name[kin_id]
        out += f" | acc_{kr}: {acc:.6f}"
    return out


@torch.no_grad()
def predict(model, val_loader, device: int | str = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset_size = len(val_loader.dataset)
    # Preallocate tensors based on the total dataset size
    similarities = torch.zeros(dataset_size, device=device)
    y_true = torch.zeros(dataset_size, dtype=torch.uint8, device=device)
    y_true_kin_relations = torch.zeros(dataset_size, dtype=torch.uint8, device=device)
    pred_kin_relations = torch.zeros(dataset_size, dtype=torch.uint8, device=device)

    current_index = 0
    for img1, img2, labels in tqdm(val_loader, total=len(val_loader), bar_format=TQDM_BAR_FORMAT):
        batch_size_current = img1.size(0)  # Handle last batch potentially being smaller
        img1, img2 = img1.to(device), img2.to(device)
        (kin_relation, is_kin) = labels
        kin_relation, is_kin = kin_relation.to(device), is_kin.to(device)

        kin, f1, f2, _ = model([img1, img2])
        sim = torch.cosine_similarity(f1, f2)

        # Fill preallocated tensors
        similarities[current_index : current_index + batch_size_current] = sim
        y_true[current_index : current_index + batch_size_current] = is_kin
        y_true_kin_relations[current_index : current_index + batch_size_current] = kin_relation
        pred_kin_relations[current_index : current_index + batch_size_current] = kin.argmax(dim=1)

        current_index += batch_size_current

    return similarities, y_true, pred_kin_relations, y_true_kin_relations


def validate(model, dataloader, device=0, threshold=None):
    model.eval()
    # Compute similarities
    similarities, y_true, pred_kin_relations, y_true_kin_relations = predict(model, dataloader)
    # Compute metrics
    auc = tm.functional.auroc(similarities, y_true, task="binary")
    fpr, tpr, thresholds = tm.functional.roc(similarities, y_true, task="binary")
    if threshold is None:
        # Get the best threshold
        maxindex = (tpr - fpr).argmax()
        threshold = thresholds[maxindex]
        if threshold.isnan().item():
            threshold = 0.01
        else:
            threshold = threshold.item()
    # Compute acc
    acc = tm.functional.accuracy(similarities, y_true, task="binary", threshold=threshold)
    # Compute accuracy with respect to kinship relations
    acc_kin_relations = {}
    for kin_relation in Sample.NAME2LABEL.values():
        mask = y_true_kin_relations == kin_relation
        acc_kin_relations[kin_relation] = tm.functional.accuracy(
            similarities[mask], y_true[mask], task="binary", threshold=threshold
        )
    kin_acc = tm.functional.accuracy(pred_kin_relations, y_true_kin_relations, task="multiclass", num_classes=12)
    return auc, threshold, acc, acc_kin_relations, kin_acc


def train(args):

    set_seed(args.seed)

    args.output_dir = Path(args.output_dir)
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write args to args.yaml
    with open(args.output_dir / "args.yaml", "w") as f:
        f.write(str(args))

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

    total_steps = len(train_loader)
    print(f"Total steps: {total_steps}")
    global_step = 0
    best_model_auc, _, val_acc, acc_kv, acc_clf_kr = validate(model, val_model_sel_loader)
    out = f"epoch: 0 | auc:  {best_model_auc:.6f} | acc_kv: {val_acc:.6f} | acc_clf_kr: {acc_clf_kr:.6f}"
    out = acc_kr_to_str(out, acc_kv)
    print(out)

    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(args.num_epoch):
        model.train()
        contrastive_loss_epoch = 0.0
        kin_loss_epoch = 0.0
        for step, data in enumerate(train_loader):
            global_step = step + epoch * args.steps_per_epoch

            image1, image2, labels = data
            (kin_relation, is_kin) = labels

            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
            kin_relation = kin_relation.to(args.device)
            is_kin = is_kin.to(args.device)

            kin, x1, x2, att = model([image1, image2])
            contrastive_loss = facornet_contrastive_loss(x1, x2, beta=att)
            kin_loss = ce_loss(kin, kin_relation)

            contrastive_loss_epoch += contrastive_loss.item()
            kin_loss_epoch += kin_loss.item()
            loss = contrastive_loss + kin_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) == args.steps_per_epoch:
                break

        use_sample = (epoch + 1) * args.batch_size * args.steps_per_epoch
        train_dataset.set_bias(use_sample)

        # Save model checkpoints
        auc, _, val_acc, acc_kv, acc_clf_kr = validate(model, val_model_sel_loader)

        if auc > best_model_auc:
            best_model_auc = auc
            torch.save(model.state_dict(), args.output_dir / "best.pth")

        out = (
            f"epoch: {epoch + 1:>2} | step: {global_step} "
            + f"| loss: {contrastive_loss_epoch / args.steps_per_epoch:.3f} "
            + f"| kin_loss: {kin_loss_epoch / args.steps_per_epoch:.3f} "
            + f"| auc: {auc:.6f} | acc_kv: {val_acc:.6f} | acc_clf_kr: {acc_clf_kr:.6f}"
        )
        out = acc_kr_to_str(out, acc_kv)
        print(out)


def val(args):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = FIW(root_dir=args.root_dir, sample_path=Path(FIW.VAL_PAIRS_THRES_SEL), transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=False)

    model = FaCoR()
    model.load_state_dict(torch.load(args.weights))
    model.to(args.device)

    auc, threshold, val_acc, acc_kr = validate(model, dataloader)
    out = f"auc:  {auc:.6f} | acc: {val_acc:.6f} | threshold: {threshold}"
    out = acc_kr_to_str(out, acc_kr)
    print(out)


def test(args):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = FIW(root_dir=args.root_dir, sample_path=Path(FIW.TEST_PAIRS), transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=False)

    model = FaCoR()
    model.load_state_dict(torch.load(args.weights))
    model.to(args.device)

    auc, threshold, val_acc, acc_kr = validate(model, dataloader, threshold=args.threshold)
    out = f"auc:  {auc:.6f} | acc: {val_acc:.6f} | threshold: {threshold}"
    out = acc_kr_to_str(out, acc_kr)
    print(out)


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
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.set_defaults(func=train)


def create_parser_val(subparsers):
    parser = subparsers.add_parser("val", help="Select best threshold for the model")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device to use for training")
    parser.set_defaults(func=val)


def create_parser_test(subparsers):
    parser = subparsers.add_parser("test", help="Test the model")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--device", type=str, default="0", help="Device to use for training")
    parser.set_defaults(func=test)


if __name__ == "__main__":

    parser = ArgumentParser(description="Configuration for the FaCoRNet strategy")
    subparsers = parser.add_subparsers()
    create_parser_train(subparsers)
    create_parser_val(subparsers)
    create_parser_test(subparsers)
    args = parser.parse_args()

    # Necessary for dataloaders?
    torch.multiprocessing.set_start_method("spawn")

    print(args)

    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.device}")
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current CUDA Device = {current_device}")
        print(f"Device Name = {device_name}")
    else:
        print("CUDA is not available.")

    args.func(args)
