from argparse import ArgumentParser
from pathlib import Path

import torch
from dataset import FIW
from model import KinshipVerifier
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import test, validate

TQDM_BAR_FORMAT = "Validating... {bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
TRAIN_PAIRS = "rfiw2021/Track1/sample0/train_sort.txt"
VAL_PAIRS_MODEL_SEL = "rfiw2021/Track1/sample0/val_choose.txt"
VAL_PAIRS_THRES_SEL = "rfiw2021/Track1/sample0/val.txt"
TEST_PAIRS = "rfiw2021/Track1/sample0/test.txt"


def contrastive_loss(x1, x2, beta=0.08):
    x1x2 = torch.cat([x1, x2], dim=0)
    x2x1 = torch.cat([x2, x1], dim=0)

    cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1), torch.unsqueeze(x1x2, dim=0), dim=2) / beta
    mask = 1.0 - torch.eye(2 * x1.size(0)).to(x1.device)
    numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / beta)
    denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
    return -torch.mean(torch.log(numerators / denominators), dim=0)


def train(args):
    # Define transformations for training and validation sets
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = FIW(args.root_dir, Path(TRAIN_PAIRS), transform=transform)
    val_model_sel_dataset = FIW(args.root_dir, Path(VAL_PAIRS_MODEL_SEL), transform=transform)
    val_thresh_sel_dataset = FIW(args.root_dir, Path(VAL_PAIRS_THRES_SEL), transform=transform)
    test_dataset = FIW(args.root_dir, Path(TEST_PAIRS), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=12, pin_memory=True)
    val_model_sel_loader = DataLoader(
        val_model_sel_dataset, batch_size=args.batch_size, num_workers=12, pin_memory=True
    )
    val_thres_sel_loader = DataLoader(
        val_thresh_sel_dataset, batch_size=args.batch_size, num_workers=12, pin_memory=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=12, pin_memory=True)

    model = KinshipVerifier(num_classes=args.num_classes, weights=args.insightface_weights, normalize=args.normalize)
    model.to(args.device)

    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    global_step = 0
    best_model_auc, _ = validate(model, val_model_sel_loader)
    print(f"epoch: 0 | auc:  {best_model_auc:.6f}")

    for epoch in range(args.num_epoch):
        model.train()
        contrastive_loss_epoch = 0
        for step, data in enumerate(train_loader):
            global_step = step + epoch * args.steps_per_epoch

            image1, image2, labels = data

            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
            labels = labels.to(args.device)

            x1 = model(image1)
            x2 = model(image2)

            loss = contrastive_loss(x1, x2, beta=args.beta)

            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

            contrastive_loss_epoch += loss.item()

            if (step + 1) == args.steps_per_epoch:
                break

        use_sample = (epoch + 1) * args.batch_size * args.steps_per_epoch
        train_dataset.set_bias(use_sample)

        # Save model checkpoints
        auc, _ = validate(model, val_model_sel_loader)

        if auc > best_model_auc:
            best_model_auc = auc
            torch.save(model.state_dict(), args.output_dir / "best.pth")

        print(
            f"epoch: {epoch + 1:>2} | step: {global_step} "
            + f"| loss: {contrastive_loss_epoch / args.steps_per_epoch:.3f} | auc: {auc:.6f}"
        )

    # best_thresh_auc, best_threshold = validate(model, val_thres_sel_loader)
    # best_acc = test(model, test_loader, best_threshold)
    # print(
    #    f"epoch: {args.num_epoch} "
    #    + f"| best_threshold_auc: {best_thresh_auc:.6f} | threshold: {best_threshold:.6f} | acc: {best_acc:.6f}"
    # )


def create_parser():
    parser = ArgumentParser(description="Configuration for the training script")

    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--train-dataset-path", type=str, required=True)
    parser.add_argument("--val-dataset-path", type=str, required=True)
    parser.add_argument("--insightface-weights", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--num-classes", type=int, default=570, help="Number of classes")
    parser.add_argument("--num-epoch", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--steps-per-epoch", type=int, default=50, help="Steps per epoch")
    parser.add_argument("--beta", type=float, default=0.08, help="Beta for contrastive loss")
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
        args.device = torch.device(f"cuda:{args.device}")
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current CUDA Device = {current_device}")
        print(f"Device Name = {device_name}")
    else:
        print("CUDA is not available.")

    train(args)