import torch
import pytest
from config import DEFAULT_PARAMS_PRETRAIN
from model import Model

def test_model_import():
    ckpt_path = 'lightning_logs/version_168/checkpoints/epoch=0-step=0.ckpt'
    state_dict = torch.load(ckpt_path, map_location='cuda')
    model = Model.load_from_checkpoint(checkpoint_path=ckpt_path, args=DEFAULT_PARAMS_PRETRAIN)
    assert True
