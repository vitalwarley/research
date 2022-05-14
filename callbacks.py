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


class MetricsCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        self.pr_curve = tm.BinnedPrecisionRecallCurve(
            num_classes=pl_module.num_classes, thresholds=100
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        target = batch[1]
        preds = outputs["preds"]

        # Precision, Recal, PR CUrve plots
        # pr and rec are a list of tensors (a tensor per class)
        # in each tensor, element i is the metric for the i threshold
        # therefore we have the shape (C, thresholds)
        pr, rec, ths = self.pr_curve(preds, target)
        pr = torch.stack(pr, dim=0).numpy()
        rec = torch.stack(rec, dim=0).numpy()

        data = pd.DataFrame(dict(precision=pr, recall=rec))
        p = sns.lineplot(data=data, x="precision", y="recall")
        plt.ylabel("precision")
        plt.xlabel("recall")
        fig = p.get_figure()
        plt.close(fig)

        self.logger.experiment.add_figure("Precision-Recall Curve", fig, batch_idx)


class EmbeddingsCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        img1, img2, label = batch
        # TODO: works for FIW with Model. Test it for LFW and others with PretrainModel.
        embs1, embs2, labels, sims = outputs.values()

        bs = len(label)

        trainer.logger.experiment.add_embedding(
            embs1,
            metadata=torch.arange(0, bs),
            label_img=img1,
            global_step=batch_idx,
            tag="embs1",
        )
        trainer.logger.experiment.add_embedding(
            embs2,
            metadata=torch.arange(0, bs),
            label_img=img2,
            global_step=batch_idx,
            tag="embs2",
        )
