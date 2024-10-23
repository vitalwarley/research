import torch
import torch.nn as nn
from components import ResidualBlock, ConvolutionBlock
class KinshipModel(nn.Module):
    def __init__(self, input_size, n_classes=2):
        super(KinshipModel, self).__init__()
        self.convblock1 = ConvolutionBlock(input_size, kernel_size=3)
        self.resblock1 = ResidualBlock(512)
        self.convblock2 = ConvolutionBlock(2, kernel_size=3)
        self.resblock2 = ResidualBlock(256)
        self.fc = nn.Linear(128, n_classes)
        # Train with nn.CrossEntropyLoss() which comes with softmax included

    def forward(self, x):
        featuresA, featuresB = x        
        featuresA = self.convblock1(featuresA)
        featuresB = self.convblock1(featuresB)

        features = torch.cat([featuresA, featuresB], dim=1) # check dimension
        features = self.convblock2(features).squeeze(1)
        features = self.resblock1(features)
        features = self.resblock2(features)
        
        output = self.fc(features)
        return output

if __name__ == "__main__":
    from base import load_pretrained_model
    from torch.utils.data import DataLoader
    from dataset import ageFIW
    import yaml
    import torch

    adaface, adaface_transform = load_pretrained_model()
    adaface.eval()

    config = yaml.safe_load(open("../params/rbkin.yml"))

    train_dataset = ageFIW(config['data_path'], "train_sort.csv", transform=adaface_transform, training=False)
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, pin_memory=False)

    images1, images2, labels = next(iter(train_loader))

    features1 = [adaface(img)[0] for img in images1]
    features2 = [adaface(img)[0] for img in images2]
    inputs1 = torch.stack(features1) # (batch_size, 5, 512)
    inputs2 = torch.stack(features2) # (batch_size, 5, 512)

    model = KinshipModel(input_size=5)
    outputs = model([inputs1, inputs2])

    breakpoint()
