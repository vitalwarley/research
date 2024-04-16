from lightning.pytorch.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add any custom arguments
        parser.add_argument("--operation:kfc:train", default=None)
        return parser


def main(args=None):
    MyLightningCLI(args=args, subclass_mode_model=True)


if __name__ == "__main__":
    main()
