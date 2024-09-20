import torch
from lightning.pytorch.cli import LightningCLI

torch.set_float32_matmul_precision("medium")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--operation:scl:train", default=None)
        return parser


def main(args=None):
    MyLightningCLI(args=args, subclass_mode_model=True)


if __name__ == "__main__":
    main()
