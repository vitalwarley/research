from argparse import ArgumentParser

import torch

from config import DEFAULT_PARAMS_PRETRAIN
from model import PretrainModel

# import pyyaml


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    model = PretrainModel.load_from_checkpoint(
        args.ckpt_path, args=DEFAULT_PARAMS_PRETRAIN
    )
    torch.save(model.state_dict(), args.output)
