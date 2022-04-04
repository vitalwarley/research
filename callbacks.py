import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pytorch_lightning.callbacks import Callback
from torchmetrics import ConfusionMatrix


class ModelInspectionCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        for name, params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(name, params, pl_module.current_epoch)


class MetricsCallback(Callback):


    def setup(self, trainer, pl_module, stage = None):
        self.cm = ConfusionMatrix(num_classes=pl_module.num_classes)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        target = batch[1]
        preds = outputs['preds']

        cm = self.cm(preds, target)
        df_cm = pd.DataFarame(cm.numpy(), index=range(pl_module.num_classes), columns=range(pl_module.num_classes))
        plt.figure(figsize=(30, 30))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()

        self.logger.experiment.add_figure("Confusion Matrix", fig_, batch_idx)
