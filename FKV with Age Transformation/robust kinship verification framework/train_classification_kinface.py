import mlflow
import yaml
from tqdm import tqdm
from dataset import ageKinFace
from model import KinshipModel
from base import load_pretrained_model

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
import torch.nn.functional as F
from torch.optim import Adam

adaface, adaface_transform = load_pretrained_model("finetuned_fiw_adaface")

def train(config, train_loader, test_loader, name):
    run_name = name
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model", "RBKin")
        mlflow.set_tag("dataset", "ageKinFace")
        mlflow.log_params(config)
        print('Starting run: ', run_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adaface.to(device)
        adaface.eval()

        model = KinshipModel(5)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
        epochs = config['epochs']

        for epoch in tqdm(range(epochs), desc="Epochs"):

            loss_epoch = 0.0
            model.train()

            for step, data in enumerate(tqdm(train_loader, desc="Training", leave=False)):
                images1, images2, labels = data

                features1 = [adaface(img.to(device))[0] for img in images1]
                features2 = [adaface(img.to(device))[0] for img in images2]
                inputs1 = torch.stack(features1)
                inputs2 = torch.stack(features2)

                outputs = model([inputs1, inputs2])
                loss = criterion(outputs, labels.to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()

            model.eval()
            with torch.no_grad():
                acc = val_model(model, test_loader)

            mlflow.log_metrics({
                "loss": loss_epoch,
                "val_accuracy": acc,
            }, step=epoch)

        mlflow.pytorch.log_model(model, "models")
        torch.save(model.state_dict(), f"models/{run_name}.pth")

        model.eval()
        with torch.no_grad():
            acc = val_model(model, test_loader)
        return acc

def val_model(model, val_loader):
    y_true = []
    y_pred = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for images1, images2, labels in val_loader:
        features1 = [adaface(img.to(device))[0] for img in images1]
        features2 = [adaface(img.to(device))[0] for img in images2]
        inputs1 = torch.stack(features1)
        inputs2 = torch.stack(features2)

        preds = model([inputs1, inputs2])
        preds = F.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)


        y_pred.extend(preds)
        y_true.extend(labels.to(device))

    y_pred = torch.stack(y_pred)
    y_true = torch.stack(y_true)
    acc = accuracy(y_pred, y_true, task="binary")
    return acc

def cross_validate(config):
    fold_means = []
    accuracies = []
    for fold in range(1, 6):
        train_dataset = ageKinFace(config['data_path'], fold, transform=adaface_transform, train=True, kinface_version=config['version'])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True)

        test_dataset = ageKinFace(config['data_path'], fold, transform=adaface_transform, kinface_version=config['version'])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True)

        fold_acc = train(config, train_loader, test_loader, f"KinFaceW-{config['version']}-fold_{fold}")
        accuracies.append(fold_acc)
        print(f"Fold {fold} accuracy: {fold_acc}")
    print(f"Mean: {sum(accuracies)/len(accuracies)}")
    return fold_means

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    exp_name = input("Type the name of the experiment: ")
    mlflow.set_experiment(exp_name)
    config =  yaml.safe_load(open("../params/rbkin.yml"))

    means = cross_validate(config)

"""     for i, folder in enumerate(['father-dau', 'father-son', 'mother-dau', 'mother-son']):
        print(f"{folder} mean accuracy: {means[i]}") """