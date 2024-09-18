import mlflow
import yaml
from tqdm import tqdm
from dataset import FIW
from base import load_pretrained_model

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

adaface, adaface_transform = load_pretrained_model()

def train(config, train_loader, test_loader):
    run_name = input('Enter run name:')
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model", "FinetunedAdaface")
        mlflow.set_tag("dataset", "FIW")
        mlflow.log_params(config) 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = adaface
        model.to(device)
        triplet_loss = nn.TripletMarginLoss()
        optimizer = Adam(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
        epochs = config['epochs']
        
        for epoch in tqdm(range(epochs), desc="Epochs"):
            
            model.train()
            train_loss = 0.0
            
            for step, data in enumerate(tqdm(train_loader, desc="Training", leave=False)):
                anchor_img, positive_img, negative_img = data

                optimizer.zero_grad()

                anchor_feature, _ = model(anchor_img.to(device))
                positive_feature, _ = model(positive_img.to(device))
                negative_feature, _ = model(negative_img.to(device))

                loss = triplet_loss(anchor_feature, positive_feature, negative_feature)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
        
            mlflow.log_metrics({
                "train_loss": train_loss / len(train_loader),
            }, step=epoch)

            mlflow.pytorch.log_model(model, "models")
        torch.save(model.state_dict(), f"models/{run_name}.pth")

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Feature Extractor")
    config = yaml.safe_load(open("../configs/fe_triplet.yml"))

    train_dataset = FIW(config['data_path'], "sample0/train_sort_triplet.csv", transform=adaface_transform, training=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=False)

    test_dataset = FIW(config['data_path'], "sample0/val.csv", transform=adaface_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=False)

    train(config, train_loader, test_loader)