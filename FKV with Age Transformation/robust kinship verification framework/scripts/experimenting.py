import sys
import os
import torch
import numpy as np
from base import load_pretrained_model
from dataset import ageFIW
sys.path.append(os.path.abspath('ours/'))
sys.path.append(os.path.abspath('ours/models'))

#from mivolo.predictor import Predictor
#from mivolo.data.data_reader import get_all_files
from tqdm import tqdm
import cv2
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

"""
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
"""

def age_to_probabilities(age_pred, age_bins, sigma=1.0):
    """
    Convert a continuous age prediction to probabilities for each age bin.
    
    Args:
    - age_pred (torch.Tensor): Predicted continuous ages (batch_size,).
    - age_bins (torch.Tensor): Age bin centers (num_bins,).
    - sigma (float): Standard deviation for the Gaussian distribution.

    Returns:
    - torch.Tensor: Probabilities for each age bin (batch_size, num_bins).
    """
    # Compute the Gaussian probabilities for each bin
    age_pred = age_pred.unsqueeze(1)  # (batch_size, 1)
    age_bins = age_bins.unsqueeze(0)  # (1, num_bins)
    prob = torch.exp(-0.5 * ((age_pred - age_bins) / sigma) ** 2)
    
    # Normalize to get probabilities
    prob = prob / prob.sum(dim=1, keepdim=True)
    return prob

# Example usage:
age_pred = torch.tensor([25.0, 35.0])  # Batch of predicted ages
age_bins = torch.tensor([10, 20, 30, 40, 50])  # Age bins

probabilities = age_to_probabilities(age_pred, age_bins)

# Testing Adaface with ageFIW
adaface, adaface_transform = load_pretrained_model()

train_dataset = ageFIW('/mnt/heavy/DeepLearning/Research/LOFAE/databases/ageFIW/Train', "train_sort_triplet.csv", transform=adaface_transform, training=True)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, pin_memory=False)

my_batch = next(iter(train_loader))

breakpoint()