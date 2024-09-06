import mlflow
import yaml
from tqdm import tqdm
from dataset import ageFIW
from model import KinshipModel
from models.base import load_pretrained_model

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

adaface, adaface_transform = load_pretrained_model()

def train(config, train_loader, test_loader):
    run_name = input('Enter run name:')
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model", "RBKin")
        mlflow.set_tag("dataset", "ageFIW")
        mlflow.log_params(config) 

        model = KinshipModel()
        
        if config['frozen_backbone']:
            adaface.eval()

        triplet_loss = nn.TripletMarginLoss()
        optimizer = Adam(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
        epochs = config['epochs']
        
        for epoch in tqdm(range(epochs), desc="Epochs"):
            
            model.train()
            train_acc = 0.0
            val_acc = 0.0
            
            for data in train_loader:
                anchor_imgs, positive_imgs, negative_imgs = data
                for i in range(config['batch_size']):
                    anchor_features, _ = adaface(anchor_imgs[i])
                    positive_features, _ = adaface(positive_imgs[i])
                    negative_features, _ = adaface(negative_imgs[i])

                    inputs = torch.stack([anchor_features, positive_features, negative_features])

                optimizer.zero_grad()

                outputs = model(inputs)
                train_acc += accuracy(outputs, labels)
                loss = triplet_loss(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs)
                val_acc += accuracy(outputs, labels)
        
            mlflow.log_metrics({
                "train_accuracy": train_acc / len(train_loader),
                "val_accuracy": val_acc / len(test_loader),
            }, step=epoch)

            mlflow.pytorch.log_model(model, "models")

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Feature Extractor")
    config = yaml.safe_load(open("../configs/rkbin.yml"))

    train_dataset = ageFIW(config['data_path'], "Train/train_sort_triplet.csv", transform=adaface_transform, training=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=False)

    test_dataset = ageFIW(config['data_path'], "Validation/val.csv", transform=adaface_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=False)

    train(config, train_loader, test_loader)