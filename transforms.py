import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import nn

_IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

_IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lightning(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        img = TF.to_tensor(img)

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        img = img.add(rgb.view(3, 1, 1).expand_as(img))

        # efficient?
        return TF.to_pil_image(img)


class ReJPGTransform(nn.Module):
    def __init__(self, prob: float = 0.5, min_quality: int = 50):
        super(ReJPGTransform, self).__init__()
        self.prob = prob
        self.min_quality = min_quality

    def forward(self, x):
        if np.random.rand() < self.prob:
            quality = np.random.randint(self.min_quality, 99)
            img = x.asnumpy()
            _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            decoded = cv2.imdecode(encoded).astype(np.float32)
            x = decoded
        return x

