import torch
from insightface.recognition.arcface_torch.backbones import get_model
from torch import nn
from torch.nn import functional as F


class InsightFace(torch.nn.Module):
    def __init__(self, num_classes, weights="", normalize: bool = False):
        super(InsightFace, self).__init__()
        self.normalize = normalize
        # Load the pre-trained backbone
        self.backbone = get_model("r100", fp16=False)
        if weights:
            self.backbone.load_state_dict(torch.load(weights))
        # Add a fully connected layer for classification
        self.fc = nn.Linear(512, num_classes)
        print("Loaded insightface model.")

        if self.normalize:
            print("Feature normalization ON.")

    def forward(self, x, return_features=False):
        # Obtain the features from the backbone
        features = self.backbone(x)
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        # Pass through the fully connected layer
        if return_features:
            return features, self.fc(features)
        return self.fc(features)


class FamilyClassifier(torch.nn.Module):
    def __init__(self, num_classes, weights="", normalize: bool = False):
        super().__init__()
        self.model = InsightFace(num_classes=num_classes, normalize=normalize)
        if weights:
            self.model.load_state_dict(torch.load(weights))

    def forward(self, x, return_features=False):
        return self.model(x, return_features=return_features)
