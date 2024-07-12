import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from components import ResidualBlock, ConvolutionBlock

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    return image

# Example: Load a dataset
image_paths = ["path_to_image1.jpg", "path_to_image2.jpg"]
images = torch.stack([load_and_preprocess_image(path) for path in image_paths])

class KinshipModel(nn.Module):
    def __init__(self, n_classes=4):
        super(KinshipModel, self).__init__()
        #self.feature_extractor = finetuned in FIW adaface
        self.convblock1 = ConvolutionBlock(3, 3, 1, 1)
        self.resblock1 = ResidualBlock(3)
        self.convblock2 = ConvolutionBlock(3, 3, 1, 1)
        self.resblock2 = ResidualBlock(3)
        self.fc = nn.Linear(3, n_classes)
        # Train with nn.CrossEntropyLoss() which comes with softmax included

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.convblock1(features)
        features = self.resblock1(features)
        features = self.convblock2(features)
        features = self.resblock2(features)
        output = self.classifier(features)
        return output
