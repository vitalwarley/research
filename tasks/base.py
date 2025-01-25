import logging
import sys
from pathlib import Path

import torch
from lightning.pytorch.cli import LightningCLI

# Add the parent directory to sys.path using pathlib
sys.path.append(str(Path(__file__).resolve().parent.parent))

torch.set_float32_matmul_precision("medium")


class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        return "available:" not in record.getMessage()


logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
logger.addFilter(IgnorePLFilter())


class BaseLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        return parser


class BaseTask:
    operation_name = None

    @classmethod
    def main(cls, args=None):
        if cls.operation_name is None:
            raise ValueError("operation_name must be set in the derived class")

        class TaskLightningCLI(BaseLightningCLI):
            def add_arguments_to_parser(self, parser):
                parser.add_argument(f"--operation:{cls.operation_name}", default=None)
                return super().add_arguments_to_parser(parser)

        TaskLightningCLI(args=args, subclass_mode_model=True)
