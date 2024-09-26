import logging

import torch
from lightning.pytorch.cli import LightningCLI

torch.set_float32_matmul_precision("medium")


class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        return "available:" not in record.getMessage()


logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
logger.addFilter(IgnorePLFilter())


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--operation:scl:train", default=None)
        # With -- prefix, the - gets converted to _
        # parser.add_argument("operation:scl:tri-subject-train", default=None)
        return parser


def main(args=None):
    MyLightningCLI(args=args, subclass_mode_model=True)


if __name__ == "__main__":
    main()
