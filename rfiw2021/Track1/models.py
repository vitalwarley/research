import os

import torch
from Track1.torch_resnet101 import *

from .insightface.recognition.arcface_torch.backbones import get_model

FILE = os.path.dirname(os.path.abspath(__file__))


class ResNet101(torch.nn.Module):
    def __init__(self, weights: str = ""):
        super(ResNet101, self).__init__()
        self.backbone = get_model("r100", fp16=False)
        if weights:
            self.backbone.load_state_dict(torch.load(weights))
        print("Loaded insightface model.")

    def forward(self, x):
        return self.backbone(x)


class Net(torch.nn.Module):
    def __init__(
        self, weights: str = "", is_insightface: bool = False, finetuned: bool = False, classification: bool = False
    ):
        super(Net, self).__init__()

        self.projection = nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )
        self.classification = classification
        if self.classification:
            self.classifier = torch.nn.Linear(256, 12)
        self._initialize_weights()

        if is_insightface:
            if finetuned:
                self.encoder = ResNet101()
            elif weights:
                self.encoder = ResNet101(weights)
            else:
                raise ValueError(f"Must provide weights or finetuned if is_insightface is True")
        else:
            self.encoder = KitModel(f"{FILE}/../backbone/kit_resnet101.pkl")
            if weights:
                self.load_state_dict(torch.load(weights))

    def _initialize_weights(self):
        modules = list(self.projection.modules())
        if self.classification:
            modules += list(self.classifier.modules())
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight - 0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        img1, img2 = imgs
        embeding1, embeding2 = self.encoder(img1), self.encoder(img2)
        pro1, pro2 = self.projection(embeding1), self.projection(embeding2)
        if self.classification:
            projs = torch.concat([pro1, pro2], dim=1)
            logits = self.classifier(projs)
            return embeding1, embeding2, pro1, pro2, logits
        else:
            return embeding1, embeding2, pro1, pro2


class NetClassifier(torch.nn.Module):
    def __init__(self, num_classes: int = 12):
        super(NetClassifier, self).__init__()
        self.encoder = KitModel(f"{FILE}/../backbone/kit_resnet101.pkl")
        self.num_classes = num_classes

        self.projection = nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )
        self.classification = nn.Sequential(
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_classes),  # number of kin relations, plus non-kin
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.projection, self.classification]:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight - 0.05, 0.05)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, images):
        # Forward function that is run when visualizing the graph
        img1, img2 = images
        embeding1, embeding2 = self.encoder(img1), self.encoder(img2)
        pro1, pro2 = self.projection(embeding1), self.projection(embeding2)
        x = torch.cat((pro1, pro2), dim=1)
        return self.classification(x)
