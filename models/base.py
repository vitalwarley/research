from collections import namedtuple
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchmetrics as tm
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR
from torchmetrics.utilities import dim_zero_cat

from datasets.utils import Sample
from models.insightface.recognition.arcface_torch.backbones import get_model
from models.utils import l2_norm

plt.switch_backend("agg")
HERE = Path(__file__).parent
models = {
    "adaface_ir_18": HERE / "../weights/adaface_ir18_webface4m.ckpt",
    "adaface_ir_50": HERE / "../weights/adaface_ir50_webface4m.ckpt",
    "adaface_ir_101": HERE
    / "../weights/adaface_ir101_webface12m.ckpt",  # "adaface_ir101_webface12m.ckpt",adaface_ir101_ms1mv3
    "adaface_ir_101_4m": HERE
    / "../weights/adaface_ir101_webface4m.ckpt",  # "adaface_ir101_webface12m.ckpt",adaface_ir101_ms1mv3
    "arcface_ir_101": HERE / "../weights/ms1mv3_arcface_r100_fp16.pth",
}


def load_pretrained_model(architecture="adaface_ir_101"):
    # load model and pretrained statedict
    assert architecture in models.keys()
    model = build_model(architecture)
    if "adaface" in architecture:
        statedict = torch.load(models[architecture], weights_only=True)["state_dict"]
        model_statedict = {
            key[6:]: val for key, val in statedict.items() if key.startswith("model.")
        }
        model.load_state_dict(model_statedict)
    else:
        model.load_state_dict(torch.load(models[architecture]))
    # model.eval()
    return model


def initialize_weights(modules):
    """Weight initilize, conv2d and linear is initialized with kaiming_normal"""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()


class Flatten(Module):
    """Flat tensor"""

    def forward(self, input):
        return input.view(input.size(0), -1)


class LinearBlock(Module):
    """Convolution block without no-linear activation layer"""

    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(
            in_c, out_c, kernel, stride, padding, groups=groups, bias=False
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SEModule(Module):
    """SE block"""

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class BasicBlockIR(Module):
    """BasicBlock for IRNet"""

    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BottleneckIR(Module):
    """BasicBlock with bottleneck for IRNet"""

    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for i in range(num_units - 1)
    ]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2),
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3),
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3),
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode="ir"):
        """Args:
        input_size: input_size of backbone
        num_layers: num_layers of backbone
        mode: support ir or irse
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [
            112,
            224,
        ], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [
            18,
            34,
            50,
            100,
            152,
            200,
        ], "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == "ir":
                unit_module = BasicBlockIR
            elif mode == "ir_se":
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == "ir":
                unit_module = BottleneckIR
            elif mode == "ir_se":
                unit_module = BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel),
                Dropout(0.4),
                Flatten(),
                Linear(output_channel * 7 * 7, 512),
                BatchNorm1d(512, affine=False),
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel),
                Dropout(0.4),
                Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False),
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        initialize_weights(self.modules())

    def forward(self, x):
        # current code only supports one extra image
        # it comes with a extra dimension for number of extra image. We will just squeeze it out for now
        x = self.input_layer(x)

        for idx, module in enumerate(self.body):
            x = module(x)
        # pdb.set_trace()
        x0 = x
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)

        return output, x0


class Backbone2(Module):
    def __init__(self, input_size, num_layers, mode="ir"):
        """Args:
        input_size: input_size of backbone
        num_layers: num_layers of backbone
        mode: support ir or irse
        """
        super(Backbone2, self).__init__()
        assert input_size[0] in [
            112,
            224,
        ], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [
            18,
            34,
            50,
            100,
            152,
            200,
        ], "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == "ir":
                unit_module = BasicBlockIR
            elif mode == "ir_se":
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == "ir":
                unit_module = BottleneckIR
            elif mode == "ir_se":
                unit_module = BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel),
                Dropout(0.4),
                Flatten(),
                Linear(output_channel * 7 * 7, 512),
                BatchNorm1d(512, affine=False),
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel),
                Dropout(0.4),
                Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False),
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        initialize_weights(self.modules())

    def forward(self, x):
        # current code only supports one extra image
        # it comes with a extra dimension for number of extra image. We will just squeeze it out for now
        x = self.input_layer(x)

        for idx, module in enumerate(self.body):
            x = module(x)
        # pdb.set_trace()
        x0 = x
        x1 = self.output_layer(x)
        # norm = torch.norm(x, 2, 1, True)
        # output = torch.div(x, norm)

        return x1, x0


def IR_18(input_size):
    """Constructs a ir-18 model."""
    model = Backbone(input_size, 18, "ir")

    return model


def IR_34(input_size):
    """Constructs a ir-34 model."""
    model = Backbone(input_size, 34, "ir")

    return model


def IR_50(input_size):
    """Constructs a ir-50 model."""
    model = Backbone(input_size, 50, "ir")

    return model


def IR_101(input_size):
    """Constructs a ir-101 model."""
    model = Backbone(input_size, 100, "ir")

    return model


def IR_101_2(input_size):
    """Constructs a ir-101 model."""
    model = Backbone2(input_size, 100, "ir")

    return model


def IR_152(input_size):
    """Constructs a ir-152 model."""
    model = Backbone(input_size, 152, "ir")

    return model


def IR_200(input_size):
    """Constructs a ir-200 model."""
    model = Backbone(input_size, 200, "ir")

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(input_size, 50, "ir_se")

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    model = Backbone(input_size, 100, "ir_se")

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    model = Backbone(input_size, 152, "ir_se")

    return model


def IR_SE_200(input_size):
    """Constructs a ir_se-200 model."""
    model = Backbone(input_size, 200, "ir_se")

    return model


def build_model(model_name="ir_50"):
    if model_name == "adaface_ir_101_4m":
        return IR_101(input_size=(112, 112))
    elif model_name == "adaface_ir_101":
        return IR_101(input_size=(112, 112))
    elif model_name == "adaface_ir_101_2":
        return IR_101_2(input_size=(112, 112))
    elif model_name == "adaface_ir_50":
        return IR_50(input_size=(112, 112))
    elif model_name == "adaface_ir_se_50":
        return IR_SE_50(input_size=(112, 112))
    elif model_name == "adaface_ir_34":
        return IR_34(input_size=(112, 112))
    elif model_name == "adaface_ir_18":
        return IR_18(input_size=(112, 112))
    elif model_name == "arcface_ir_101":
        return get_model("r100", fp16=True)
    else:
        raise ValueError("not a correct model name", model_name)


class CollectPreds(tm.Metric):
    def __init__(self, name: str, **kwargs):
        self.name = name
        super().__init__(**kwargs)
        self.add_state("predictions", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor):
        # Append current batch predictions to the list of all predictions
        self.predictions.append(preds)

    def compute(self):
        # Concatenate the list of predictions into a single tensor
        return dim_zero_cat(self.predictions)


class LightningBaseModel(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: list[torch.nn.Module] | torch.nn.Module,
        optimizer="SGD",
        adamw_beta1=0.9,
        adamw_beta2=0.999,
        lr=1e-4,
        momentum=0.9,
        weight_decay=0,
        start_lr=1e-4,
        end_lr=1e-10,
        lr_factor=0.75,
        lr_steps=[8, 14, 25, 35, 40, 50, 60],
        warmup=200,
        cooldown=400,
        scheduler=None,
        anneal_strategy="cos",
        threshold=None,
        # TODO: how to add the below params only to the subclass?
        weights=None,
        list_dir="",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model"))

        self.model = model
        self.criterion = loss
        self.threshold = threshold
        self.list_dir = list_dir

        self.similarities = CollectPreds("similarities")
        self.is_kin_labels = CollectPreds("is_kin_labels")
        self.kin_labels = CollectPreds("kin_labels")

        self.sample_cls = Sample

        if weights:
            print(f"Loading weights from {weights}")
            self.load_weights(weights)

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, weights_only=False
        )  # TODO: weights_only=True
        # Only load model weights, not optimizer state etc.
        model_state = {
            k: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")
        }
        self.model.load_state_dict(model_state, strict=False)

    def forward(self, inputs):
        return self.model(inputs)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        print(f"Total steps = {self.trainer.estimated_stepping_batches}")
        if self.hparams.scheduler is None:
            self.hparams.start_lr = self.hparams.lr

        if self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.start_lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.start_lr,
                betas=(self.hparams.adamw_beta1, self.hparams.adamw_beta2),
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")

        config = {
            "optimizer": optimizer,
        }

        # FIXME: improve -- add as class_path
        if self.hparams.scheduler == "multistep":
            config["lr_scheduler"] = {
                "scheduler": MultiStepLR(
                    optimizer,
                    milestones=self.hparams.lr_steps,
                    gamma=self.hparams.lr_factor,
                ),
            }
        # OneCycleLR with warmup and cooldown
        elif self.hparams.scheduler == "onecycle":
            config["lr_scheduler"] = {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=self.hparams.lr,
                    total_steps=self.trainer.estimated_stepping_batches,
                    pct_start=0.3,  # Assume 30% of the time to reach the peak (typical configuration)
                    div_factor=self.hparams.lr / self.hparams.start_lr,
                    final_div_factor=self.hparams.end_lr / self.hparams.start_lr,
                    anneal_strategy=self.hparams.anneal_strategy,
                ),
            }

        print(f"optimizers config = {config}")
        # LOGGER.info("Model will train for steps=%s", self.trainer.estimated_stepping_batches)
        return config

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        if not self.hparams.scheduler == "onecycle":
            # warm up lr
            if self.trainer.global_step < self.hparams.warmup:
                # print formula below
                cur_lr = (self.trainer.global_step + 1) * (
                    self.hparams.lr - self.hparams.start_lr
                ) / self.hparams.warmup + self.hparams.start_lr
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
            # cool down lr
            elif (
                self.trainer.global_step
                > self.trainer.estimated_stepping_batches - self.hparams.cooldown
            ):  # cooldown start
                cur_lr = (
                    self.trainer.estimated_stepping_batches - self.trainer.global_step
                ) * (
                    optimizer.param_groups[0]["lr"] - self.hparams.end_lr
                ) / self.hparams.cooldown + self.hparams.end_lr
                optimizer.param_groups[0]["lr"] = cur_lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def on_train_epoch_end(self):
        # Calculate the number of samples processed
        use_sample = (
            (self.current_epoch + 1)
            * self.trainer.datamodule.batch_size
            * int(self.trainer.limit_train_batches)
        )
        # Update the dataset's bias or sampling strategy
        if hasattr(self.trainer.datamodule.train_dataset, "set_bias"):
            self.trainer.datamodule.train_dataset.set_bias(use_sample)

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")

    def _compute_collections(self):
        similarities = self.similarities.compute()
        is_kin_labels = self.is_kin_labels.compute()
        kin_labels = self.kin_labels.compute()
        return similarities, is_kin_labels, kin_labels

    def _reset_collections(self):
        self.similarities.reset()
        self.is_kin_labels.reset()
        self.kin_labels.reset()

    def _on_epoch_end(self, stage):
        self._compute_metrics(stage=stage)
        # Reset predictions
        self._reset_collections()

    def _compute_best_threshold(self, tpr, fpr, thresholds, stage="val"):
        maxindex = (tpr - fpr).argmax()

        # Compute best threshold
        if stage == "test" and self.threshold is None:
            raise ValueError("Threshold must be provided for test stage")
        elif stage == "test":
            best_threshold = self.threshold
        else:  # Compute best threshold for training or validation
            best_threshold = thresholds[maxindex]  # probability
            # Check if is nan, if so, set to 0.5
            if torch.isnan(best_threshold):
                best_threshold = 0.5
            else:
                best_threshold = best_threshold.item()

        # Compute metrics
        #   - similarities will be converted to probabilites,
        #   - therefore best_threshold must be a probability
        if stage == "test":
            best_threshold = torch.sigmoid(torch.tensor(best_threshold)).item()
            # val stage computes its own threshold, which is already a probability

        return best_threshold, maxindex

    def _compute_metrics(self, stage="train"):
        similarities, is_kin_labels, kin_labels = self._compute_collections()

        # Compute basic metrics
        basic_metrics = self._compute_basic_metrics(similarities, is_kin_labels, stage)
        fpr, tpr, maxindex, threshold, accuracy, auc, precision, recall, f1 = (
            basic_metrics.values()
        )

        self._compute_kinship_metrics(
            similarities, is_kin_labels, kin_labels, threshold
        )

        threshold = torch.logit(torch.tensor(threshold))
        self._plot_roc_curve_and_histogram(
            auc, fpr, tpr, maxindex, similarities, is_kin_labels, threshold
        )

        # Log basic metrics
        metrics = zip(
            ["threshold", "accuracy", "auc", "precision", "recall", "f1"],
            [threshold, accuracy, auc, precision, recall, f1],
        )
        for metric_name, metric_value in metrics:
            self.log(
                f"{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def _compute_basic_metrics(self, similarities, is_kin_labels, stage):
        fpr, tpr, thresholds = tm.functional.roc(
            similarities, is_kin_labels, task="binary"
        )
        best_threshold, maxindex = self._compute_best_threshold(
            tpr, fpr, thresholds, stage
        )

        return {
            "fpr": fpr,
            "tpr": tpr,
            "maxindex": maxindex,
            "threshold": best_threshold,
            "accuracy": tm.functional.accuracy(
                similarities, is_kin_labels, threshold=best_threshold, task="binary"
            ),
            "auc": tm.functional.auroc(similarities, is_kin_labels, task="binary"),
            "precision": tm.functional.precision(
                similarities, is_kin_labels, threshold=best_threshold, task="binary"
            ),
            "recall": tm.functional.recall(
                similarities, is_kin_labels, threshold=best_threshold, task="binary"
            ),
            "f1": tm.functional.f1_score(
                similarities, is_kin_labels, threshold=best_threshold, task="binary"
            ),
        }

    def _compute_kinship_metrics(
        self, similarities, is_kin_labels, kin_labels, threshold
    ):
        for kin, kin_id in self.sample_cls.NAME2LABEL.items():
            mask = kin_labels == kin_id
            if torch.any(mask):
                acc = tm.functional.accuracy(
                    similarities[mask],
                    is_kin_labels[mask].int(),
                    threshold=threshold,
                    task="binary",
                )
                self.log(
                    f"accuracy/{kin}",
                    acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

                self._log_kinship_similarities(
                    similarities[mask], is_kin_labels[mask], kin
                )

    def _log_kinship_similarities(self, similarities, is_kin_labels, kin):
        positives = similarities[is_kin_labels == 1]
        negatives = similarities[is_kin_labels == 0]
        if positives.numel() > 0:
            self.logger.experiment.add_histogram(
                f"similarities/positives/{kin}",
                positives,
                global_step=self.current_epoch,
            )
        if negatives.numel() > 0:
            self.logger.experiment.add_histogram(
                f"similarities/negatives/{kin}",
                negatives,
                global_step=self.current_epoch,
            )

    def _plot_roc_curve_and_histogram(
        self, auc, fpr, tpr, maxindex, similarities, is_kin_labels, best_threshold
    ):
        # Convert to numpy
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        similarities = similarities.cpu().numpy()
        is_kin_labels = is_kin_labels.cpu().numpy()
        best_threshold = best_threshold.cpu().numpy()

        # Plot ROC Curve
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].plot(
            fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % auc
        )
        axs[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axs[0].scatter(
            fpr[maxindex],
            tpr[maxindex],
            s=50,
            c="red",
            label=f"Threshold ({best_threshold:.6f})",
        )
        axs[0].set_xlim([0.0, 1.0])
        axs[0].set_ylim([0.0, 1.05])
        axs[0].set_xlabel("False Positive Rate")
        axs[0].set_ylabel("True Positive Rate")
        axs[0].set_title("Receiver Operating Characteristic")
        axs[0].legend(loc="lower right")

        # Plot Histogram of Similarities
        positives = [
            similarities[i] for i in range(len(similarities)) if is_kin_labels[i] == 1
        ]
        negatives = [
            similarities[i] for i in range(len(similarities)) if is_kin_labels[i] == 0
        ]

        axs[1].hist(positives, bins=20, alpha=0.5, label="Positive", color="g")
        axs[1].hist(negatives, bins=20, alpha=0.5, label="Negative", color="r")
        axs[1].axvline(
            x=best_threshold,
            color="b",
            linestyle="--",
            label=f"Threshold ({best_threshold:.6f})",
        )
        axs[1].set_xlabel("Similarity")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Histogram of Similarities")
        axs[1].legend(loc="upper right")

        plt.tight_layout()

        self.logger.experiment.add_figure(
            "ROC Curve and Histogram of Similarities",
            fig,
            global_step=self.current_epoch,
        )
        plt.close(fig)
        plt.close("all")


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)

        initialize_weights(self.modules())

    def forward(self, x):
        x = self.layer1(x)
        return x


class SimpleModel(torch.nn.Module):
    def __init__(self, model: str, projection: None | tuple):
        super().__init__()
        self.model = model
        self.backbone = load_pretrained_model(model)
        self.projection = projection
        if self.projection:
            self.projection_head = ProjectionHead(*projection)

    def forward(self, imgs):
        img1, img2 = imgs
        idx = [2, 1, 0]

        if "adaface" in self.model:
            z_0, x1_feat = self.backbone(img1[:, idx])  # (B, 512) and (B, 512, 7, 7)
            z_1, x2_feat = self.backbone(img2[:, idx])  # ...
        else:
            z_0 = self.backbone(img1)
            z_1 = self.backbone(img2)

        if self.projection:
            z_0 = self.projection_head(z_0)
            z_1 = self.projection_head(z_1)

        z_0 = l2_norm(z_0)
        z_1 = l2_norm(z_1)

        return z_0, z_1


class SimpleModelTask2(SimpleModel):
    def forward(self, imgs):
        img1, img2, img3 = imgs
        idx = [2, 1, 0]

        if "adaface" in self.model:
            z_1, x1_feat = self.backbone(img1[:, idx])  # (B, 512) and (B, 512, 7, 7)
            z_2, x2_feat = self.backbone(img2[:, idx])  # ...
            z_3, x3_feat = self.backbone(img3[:, idx])  # ...
        else:
            z_1 = self.backbone(img1)
            z_2 = self.backbone(img2)
            z_3 = self.backbone(img3)

        if self.projection:
            z_1 = self.projection_head(z_1)
            z_2 = self.projection_head(z_2)
            z_3 = self.projection_head(z_3)

        z_1 = l2_norm(z_1)
        z_2 = l2_norm(z_2)
        z_3 = l2_norm(z_3)

        return z_1, z_2, z_3
