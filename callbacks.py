import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics as tm
from pytorch_lightning.callbacks import Callback


class ModelInspectionCallback(Callback):
    def on_test_start(self, trainer, pl_module):
        for name, params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(
                name, params, pl_module.current_epoch
            )
        import sys

        sys.exit()
