from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from models.facornet import FaCoRNetLightning

from datasets.facornet import FaCoRNetDataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add custom arguments or modify existing ones
        parser.add_argument("--threshold", type=float, default=None, help="Threshold for classification")

        # Example of adding a callback argument
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")

    def before_fit(self):
        # Custom code before the fit starts, for example:
        if self.config["threshold"] is not None:
            self.model.threshold = self.config["threshold"]


def main():
    cli = MyLightningCLI(FaCoRNetLightning, FaCoRNetDataModule)
    cli.run()


if __name__ == "__main__":
    main()
