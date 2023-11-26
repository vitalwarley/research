import torch
from insightface.recognition.arcface_torch.backbones import get_model
from torch import nn
from torch.nn import functional as F


class InsightFace(torch.nn.Module):
    def __init__(self, num_classes, weights="", normalize: bool = False):
        super(InsightFace, self).__init__()
        self.normalize = normalize
        # Load the pre-trained backbone
        self.backbone = get_model("r100", fp16=True)
        if weights:
            print("Loaded insightface model.")
            self.backbone.load_state_dict(torch.load(weights))
        # Add a fully connected layer for classification
        # TODO: fp16 here how?
        self.fc = nn.Linear(512, num_classes)
        torch.nn.init.normal_(self.fc.weight, std=0.01)

        if self.normalize:
            print("Feature normalization ON.")

    def forward(self, x, return_features=False):
        # Obtain the features from the backbone
        features = self.backbone(x)
        if self.normalize:
            features = F.normalize(features, p=2, dim=1) * 32  # Reproduce ArcFace scale s
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


class KinshipVerifier(nn.Module):
    def __init__(self, num_classes: int, weights: str = "", normalize: bool = False):
        super().__init__()
        self.normalize = normalize

        if weights:
            if "ms1mv3" in weights:
                self.model = InsightFace(num_classes=num_classes, weights=weights, normalize=normalize)
            else:
                self.model = InsightFace(num_classes=num_classes, normalize=normalize)
                self.model.load_state_dict(torch.load(weights))
        else:
            self.model = InsightFace(num_classes=num_classes, normalize=normalize)

        # Drops fc layer
        self.model = self.model.backbone

        self.projection = nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight - 0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        # Obtain the features from the backbone
        features = self.model(x)
        if self.normalize:
            features = F.normalize(features, p=2, dim=1) * 32  # Reproduce ArcFace scale s
        # Pass through the fully connected layer
        if return_features:
            return features, self.projection(features)
        return self.projection(features)
