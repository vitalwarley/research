import argparse

import cv2
import numpy as np
import torch

from insightface.recognition.arcface_torch.backbones import get_model
from insightface.recognition.arcface_torch.eval import verification


def validate(weight, name, bin, image_size):

    dataset = verification.load_bin(bin, (image_size, image_size))
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
    net.eval()
    net.cuda()
    _, _, acc, std, norm, _ = verification.test(dataset, net, batch_size=256, nfolds=10)
    print('Accuracy: %1.3f+-%1.3f' % (acc, std))
    print('Norm: %1.3f' % (norm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--bin', type=str, default=None)
    parser.add_argument('--imgsz', type=int)
    args = parser.parse_args()
    validate(args.weight, args.network, args.bin, args.imgsz)
