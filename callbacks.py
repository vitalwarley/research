import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchmetrics as tm
from pytorch_lightning.callbacks import Callback


class ModelInspectionCallback(Callback):
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ):
        for name, params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(
                name, params, pl_module.current_epoch
            )
