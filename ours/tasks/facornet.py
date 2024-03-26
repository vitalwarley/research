from lightning.pytorch.cli import LightningCLI


def main(args=None):
    cli = LightningCLI(args=args, subclass_mode_model=True)  # noqa: F841


if __name__ == "__main__":
    main()
