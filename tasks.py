from torch import utils
from torchvision import transforms, datasets

import transforms as mytransforms
from dataset import FamiliesDataset, FamiliesDataModule, MS1MDataModule


def init_cifar(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = datasets.CIFAR10(
        root="~/datasets", train=True, download=True, transform=transform
    )
    trainloader = utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    valset = datasets.CIFAR10(
        root="~/datasets", train=False, download=True, transform=transform
    )
    valloader = utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainloader, valloader


def init_ms1m(args):
    train_transforms = transforms.Compose(
        [
            # transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    datamodule = MS1MDataModule(
        args.num_classes,
        args.data_dir,
        transforms=[train_transforms, val_transforms],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return datamodule


def init_fiw(args):
    train_transforms = transforms.Compose(
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
            # ReJPGTransform(0.3, 70),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose([transforms.ToTensor()])

    # add params from args
    datamodule = FamiliesDataModule(
        args.data_dir,
        transforms=[train_transforms, val_transforms],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return datamodule
