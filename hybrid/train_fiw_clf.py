from argparse import ArgumentParser
from pathlib import Path

import torch
from dataset import FamiliesDataset
from model import InsightFace
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


def train(args):
    # Define transformations for training and validation sets
    transform_img_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # More transformations can be added here
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the training dataset
    train_dataset = FamiliesDataset(Path(args.dataset_path), transform=transform_img_train)
    num_classes = len(train_dataset.families)  # This should be the number of families or classes

    # Define the model
    model = InsightFace(num_classes=num_classes, weights=args.insightface_weights)
    model.to(device)

    # Define the DataLoader for the training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=48,  # Assuming a batch size of 48 as in the original script
        shuffle=True,
        num_workers=12,  # Assuming 12 workers for loading data
        pin_memory=True,
    )

    # Define the optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(20):  # Assuming 20 epochs as in the original script
        model.train()
        running_loss = 0.0
        for i, (img, family_idx, person_idx) in enumerate(train_loader):
            # Transfer to GPU if available
            inputs, labels = img.to(device), family_idx.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_function(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print("[Epoch: %d, Mini-batch: %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Save model checkpoints
        if epoch % 10 == 9:  # Save every 10 epochs
            torch.save(model.state_dict(), f"{args.output_dir}/model_epoch_{epoch + 1}.pth")

    print("Finished Training")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--insightface-weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    train(args)
