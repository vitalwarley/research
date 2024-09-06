import mlflow
import yaml
from tqdm import tqdm
from dataset import ageFIW
from model import KinshipModel
from base import load_pretrained_model

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torch.optim import Adam

adaface, adaface_transform = load_pretrained_model()

def train(config, train_loader, test_loader):
    run_name = input('Enter run name:')
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model", "RBKin")
        mlflow.set_tag("dataset", "ageFIW")
        mlflow.log_params(config) 

        model = KinshipModel()

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
        epochs = config['epochs']
        
        for epoch in tqdm(range(epochs), desc="Epochs"):
            
            loss_epoch = 0.0
            model.train()

            for data in train_loader:
                images1, images2, labels = data
                
                features1 = [adaface(img)[0] for img in images1]
                features2 = [adaface(img)[0] for img in images2]
                inputs1 = torch.stack(features1)
                inputs2 = torch.stack(features2)

                outputs = model([inputs1, inputs2])
                loss = criterion(outputs, labels)
                
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

def val_model(model, val_loader):
    y_true = []
    y_pred = []
    for img1, img2, labels in val_loader:
        preds = model([img1.cuda(), img2.cuda()])
        y_pred.extend(preds)
        y_true.extend(labels)
    y_pred = torch.stack(y_pred)
    y_true = torch.stack(y_true)
    acc = accuracy(y_pred, y_true, task="multiclass", num_classes=model.num_classes)
    return acc

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("RBKin")
    config = yaml.safe_load(open("../configs/rkbin.yml"))

    train_dataset = ageFIW(config['data_path'], "Train/train_sort_triplet.csv", transform=adaface_transform, training=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=False)

    test_dataset = ageFIW(config['data_path'], "Validation/val.csv", transform=adaface_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=False)

    train(config, train_loader, test_loader)