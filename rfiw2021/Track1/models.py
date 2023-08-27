import os

import torch
from Track1.torch_resnet101 import *

FILE = os.path.dirname(os.path.abspath(__file__))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = KitModel(f"{FILE}/../backbone/kit_resnet101.pkl")

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

    def forward(self, imgs):
        img1, img2 = imgs
        embeding1, embeding2 = self.encoder(img1), self.encoder(img2)
        pro1, pro2 = self.projection(embeding1), self.projection(embeding2)
        return embeding1, embeding2, pro1, pro2


class NetClassifier(torch.nn.Module):
    def __init__(self):
        super(NetClassifier, self).__init__()
        self.encoder = KitModel(f"{FILE}/../backbone/kit_resnet101.pkl")

        self.projection = nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )
        self.classification = nn.Sequential(
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 12),  # number of kin relations, plus non-kin
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
