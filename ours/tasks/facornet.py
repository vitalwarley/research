from lightning.pytorch.cli import LightningCLI


def main(args=None):
    LightningCLI(args=args, subclass_mode_model=True)

    
if __name__ == "__main__":
    main()
