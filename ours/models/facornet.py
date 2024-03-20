from collections import namedtuple
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torchmetrics as tm
from datasets.utils import Sample
from losses import facornet_contrastive_loss
from models.utils import compute_best_threshold
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

# Assuming the necessary imports are done for FaCoR, facornet_contrastive_loss, FIW, and other utilities

HERE = Path(__file__).parent

adaface_models = {
    "ir_50": "pretrained/adaface_ir50_ms1mv2.ckpt",
    "ir_101": HERE / "../weights/adaface_ir101_webface12m.ckpt",  # "adaface_ir101_webface12m.ckpt",adaface_ir101_ms1mv3
    "ir_101_2": "adaface_ir101_webface12m.ckpt",  # "adaface_ir101_webface12m.ckpt",adaface_ir101_ms1mv3
}


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def load_pretrained_model(architecture="ir_101"):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = build_model(architecture)
    statedict = torch.load(adaface_models[architecture])["state_dict"]
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith("model.")}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor


class FaCoR(torch.nn.Module):
    def __init__(self):
        super(FaCoR, self).__init__()
        self.backbone = load_pretrained_model("ir_101")
        self.projection = nn.Sequential(
            torch.nn.Linear(512 * 6, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )
        self.channel = 64
        self.spatial_ca = SpatialCrossAttention(self.channel * 8, CA=True)
        self.channel_ca = ChannelCrossAttention(self.channel * 8)
        self.CCA = ChannelInteraction(1024)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight - 0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs, aug=False):
        img1, img2 = imgs
        idx = [2, 1, 0]
        f1_0, x1_feat = self.backbone(img1[:, idx])
        f2_0, x2_feat = self.backbone(img2[:, idx])

        _, _, att_map0 = self.spatial_ca(x1_feat, x2_feat)

        f1_0 = l2_norm(f1_0)
        f2_0 = l2_norm(f2_0)

        x1_feat = l2_norm(x1_feat)
        x2_feat = l2_norm(x2_feat)

        f1_1, f2_1, _ = self.channel_ca(f1_0, f2_0)
        f1_2, f2_2, _ = self.spatial_ca(x1_feat, x2_feat)

        f1_2 = torch.flatten(self.avg_pool(f1_2), 1)
        f2_2 = torch.flatten(self.avg_pool(f2_2), 1)

        wC = self.CCA(torch.cat([f1_1, f1_2], 1).unsqueeze(2).unsqueeze(3))
        wC = wC.view(-1, 2, 512)[:, :, :, None, None]
        f1s = f1_1.unsqueeze(2).unsqueeze(3) * wC[:, 0] + f1_2.unsqueeze(2).unsqueeze(3) * wC[:, 1]

        wC2 = self.CCA(torch.cat([f2_1, f2_2], 1).unsqueeze(2).unsqueeze(3))
        wC2 = wC2.view(-1, 2, 512)[:, :, :, None, None]
        f2s = f2_1.unsqueeze(2).unsqueeze(3) * wC2[:, 0] + f2_2.unsqueeze(2).unsqueeze(3) * wC2[:, 1]

        f1s = torch.flatten(f1s, 1)
        f2s = torch.flatten(f2s, 1)

        return f1s, f2s, att_map0


class SpatialCrossAttention(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_dim, CA=False):
        super(SpatialCrossAttention, self).__init__()
        self.chanel_in = in_dim
        reduction_ratio = 16
        self.query_conv = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim // reduction_ratio, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.CA = ChannelInteraction(in_dim * 2)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.UseCA = CA
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x1, x2):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x1.size()
        x = torch.cat([x1, x2], 1)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value1 = self.value_conv1(x1)
        proj_value2 = self.value_conv2(x2)

        proj_value1 = proj_value1.view(m_batchsize, -1, width * height)  # B X C X N
        proj_value2 = proj_value2.view(m_batchsize, -1, width * height)  # B X C X N

        out1 = torch.bmm(proj_value1, attention.permute(0, 2, 1))
        out2 = torch.bmm(proj_value2, attention.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, -1, width, height) + x1.view(m_batchsize, -1, width, height)
        out2 = out2.view(m_batchsize, -1, width, height) + x2.view(m_batchsize, -1, width, height)
        # out = self.gamma*out + x.view(m_batchsize,2*C,width,height)
        return out1, out2, attention


class ChannelInteraction(nn.Module):
    def __init__(self, channel):
        super(ChannelInteraction, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # y = self.avg_pool(x)
        y = x
        # pdb.set_trace()
        y = self.ca(y)
        return y


class ChannelCrossAttention(nn.Module):
    """Channel attention module"""

    def __init__(self, in_dim):
        super(ChannelCrossAttention, self).__init__()
        self.chanel_in = in_dim

        self.conv = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X C X C
        """
        x1 = x1.unsqueeze(2).unsqueeze(3)
        x2 = x2.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x1, x2], 1)
        x = self.conv(x)
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        proj_value1 = x1.view(m_batchsize, C, -1)
        proj_value2 = x2.view(m_batchsize, C, -1)

        out1 = torch.bmm(attention, proj_value1)
        out1 = out1.view(m_batchsize, C, height, width)

        out2 = torch.bmm(attention, proj_value2)
        out2 = out2.view(m_batchsize, C, height, width)

        out1 = out1 + x1
        out2 = out2 + x2
        # out = self.gamma*out + x
        return out1.reshape(m_batchsize, -1), out2.reshape(m_batchsize, -1), attention


def build_model(model_name="ir_50"):
    if model_name == "ir_101":
        return IR_101(input_size=(112, 112))
    elif model_name == "ir_101_2":
        return IR_101_2(input_size=(112, 112))
    elif model_name == "ir_50":
        return IR_50(input_size=(112, 112))
    elif model_name == "ir_se_50":
        return IR_SE_50(input_size=(112, 112))
    elif model_name == "ir_34":
        return IR_34(input_size=(112, 112))
    elif model_name == "ir_18":
        return IR_18(input_size=(112, 112))
    else:
        raise ValueError("not a correct model name", model_name)


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

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
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
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)

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
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
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
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
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

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


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
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
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
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
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
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
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
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
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


class DynamicThresholdAccuracy(tm.Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def __call__(self, preds: torch.Tensor, target: torch.Tensor, threshold: torch.Tensor):
        self.update(preds, target, threshold)
        return self.compute()

    def update(self, preds: torch.Tensor, target: torch.Tensor, threshold: torch.Tensor):
        preds_thresholded = preds >= threshold.unsqueeze(
            1
        )  # Assuming threshold is a 1D tensor with the same batch size as preds
        correct = torch.sum(preds_thresholded == target)
        self.correct += correct
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class CollectPreds(tm.Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("predictions", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor):
        # Convert preds to the same device as the metric state
        preds = preds.detach().to(self.predictions[0].device if self.predictions else preds.device)

        # Append current batch predictions to the list of all predictions
        self.predictions.append(preds)

    def compute(self):
        # Concatenate the list of predictions into a single tensor
        return torch.cat(self.predictions, dim=0)

    def reset(self):
        # Reset the state (list of predictions)
        self.predictions = []


class FaCoRNetLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, momentum=0.9, weight_decay=0, weights_path=None, threshold=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = FaCoR()

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.loss_fn = facornet_contrastive_loss

        self.threshold = threshold

        self.similarities = CollectPreds()  # Custom metric to collect predictions
        self.is_kin_labels = CollectPreds()  # Custom metric to collect labels
        self.kin_labels = CollectPreds()  # Custom metric to collect labels

        # Metrics
        self.train_auc = tm.AUROC(task="binary")
        self.val_auc = tm.AUROC(task="binary")
        self.train_acc = DynamicThresholdAccuracy()
        self.val_acc = DynamicThresholdAccuracy()
        self.train_acc_kin_relations = tm.MetricCollection(
            {f"train/acc_{kin}": DynamicThresholdAccuracy() for kin in Sample.NAME2LABEL.values()}
        )
        self.val_acc_kin_relations = tm.MetricCollection(
            {f"val/acc_{kin}": DynamicThresholdAccuracy() for kin in Sample.NAME2LABEL.values()}
        )

    def setup(self, stage):
        # TODO: use checkpoint callback to load the weights
        if self.hparams.weights_path is not None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                # Load the weights
                state_dict = torch.load(self.hparams.weights_path, map_location=map_location)
                self.model.load_state_dict(state_dict)
                print(f"Loaded weights from {self.hparams.weights_path}")
            except FileNotFoundError:
                print(f"Failed to load weights from {self.hparams.weights_path}. File does not exist.")
            except RuntimeError as e:
                print(f"Failed to load weights due to a runtime error: {e}")

    def forward(self, img1, img2):
        return self.model([img1, img2])

    def step(self, batch, stage="train"):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        f1, f2, att = self.forward(img1, img2)
        loss = self.loss_fn(f1, f2, beta=att)
        sim = torch.cosine_similarity(f1, f2)

        if stage == "train":
            self.__compute_metrics(sim, is_kin.int(), kin_relation, stage)
        else:
            # Compute best threshold for training or validation
            self.similarities(sim)
            self.is_kin_labels(is_kin.int())
            self.kin_labels(kin_relation)

        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        return optimizer

    def on_epoch_end(self):
        # Calculate the number of samples processed
        use_sample = (self.current_epoch + 1) * self.trainer.datamodule.batch_size * self.trainer.limit_train_batches
        # Update the dataset's bias or sampling strategy
        self.trainer.datamodule.train_dataset.set_bias(use_sample)
        # Reset the metrics
        self.similarities.reset()
        self.is_kin_labels.reset()
        self.kin_labels.reset()

    def on_validation_epoch_end(self, outputs):
        similarities = self.similarities.compute()
        is_kin_labels = self.is_kin_labels.compute()
        kin_labels = self.kin_labels.compute()
        self.__compute_metrics(similarities, is_kin_labels, kin_labels, stage="val")

    def on_test_epoch_end(self):
        similarities = self.similarities.compute()
        is_kin_labels = self.is_kin_labels.compute()
        kin_labels = self.kin_labels.compute()
        self.__compute_metrics(similarities, is_kin_labels, kin_labels, stage="test")

    def __compute_metrics(self, similarities, is_kin_labels, kin_labels, stage="train"):
        if stage == "test" and self.threshold is None:
            raise ValueError("Threshold must be provided for test stage")
        elif stage == "test":
            best_threshold = self.threshold
        else:  # Compute best threshold for training or validation
            fpr, tpr, thresholds = tm.functional.roc(similarities, is_kin_labels, task="binary")
            best_threshold = compute_best_threshold(tpr, fpr, thresholds)
        self.log(f"{stage}/threshold", best_threshold, on_epoch=True, prog_bar=True, logger=True)

        # Log AUC and Accuracy
        auc_fn = self.train_auc if stage == "train" else self.val_auc
        acc_fn = self.train_acc if stage == "train" else self.val_acc
        auc = auc_fn(similarities, is_kin_labels, best_threshold)
        acc = acc_fn(similarities, is_kin_labels, best_threshold)
        self.log(f"{stage}/auc", auc, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}/acc", acc, on_epoch=True, prog_bar=True, logger=True)

        # Accuracy for each kinship relation
        acc_kin_relations = self.train_acc_kin_relations if stage == "train" else self.val_acc_kin_relations
        for kin_id in Sample.NAME2LABEL.values():
            mask = kin_labels == kin_id
            if torch.any(mask):
                acc_kin_relations[f"val/acc_{kin_id}"](similarities[mask], is_kin_labels[mask].int(), best_threshold)
                self.log(
                    f"{stage}/acc_{kin_id}",
                    acc_kin_relations[f"val/acc_{kin_id}"],
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )


if __name__ == "__main__":
    model = IR_101((112, 112))
    print(model)
    input = torch.rand(2, 3, 112, 112)
    output = model(input)
    print(output)
