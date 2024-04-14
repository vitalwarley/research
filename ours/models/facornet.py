from collections import namedtuple
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics as tm
from datasets.utils import Sample, SampleKFC
from losses import contrastive_loss, facornet_contrastive_loss
from pytorch_metric_learning.losses import ArcFaceLoss
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
    # model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor


class HeadKin(nn.Module):
    def __init__(self, in_features=512, out_features=4, ratio=8):
        super().__init__()
        self.projection_head = nn.Sequential(
            # TODO: think better
            torch.nn.Linear(2 * in_features, in_features // ratio),
            torch.nn.BatchNorm1d(in_features // ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // ratio, out_features),
        )

        self.initialize_weights(self.projection_head)

    def initialize_weights(self, proj_head):
        for m in proj_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight - 0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

    def forward(self, em):
        return self.projection_head(em)


class HeadFamily(nn.Module):
    def __init__(self, in_features=512, out_features=4, ratio=2):
        super().__init__()
        self.projection_head = nn.Sequential(
            torch.nn.Linear(in_features, in_features // ratio),
            torch.nn.BatchNorm1d(in_features // ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // ratio, out_features),
        )

    def forward(self, em):
        return self.projection_head(em)


class FaCoR(torch.nn.Module):
    def __init__(self):
        super(FaCoR, self).__init__()
        self.backbone = load_pretrained_model("ir_101")
        self.channel = 64
        self.spatial_ca = SpatialCrossAttention(self.channel * 8, CA=True)
        self.channel_ca = ChannelCrossAttention(self.channel * 8)
        self.CCA = ChannelInteraction(1024)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.task_kin = HeadKin(512, 12, 8)

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

        # fc = torch.cat([f1s, f2s], dim=1)
        # kin = self.task_kin(fc)

        # return kin, f1s, f2s, att_map0
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


class FaCoRNetLightning(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
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
        num_families=0,
        loss_factor=0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model"))

        self.model = FaCoR() or model
        self.loss_fn = facornet_contrastive_loss
        self.threshold = threshold

        self.similarities = CollectPreds("similarities")
        self.is_kin_labels = CollectPreds("is_kin_labels")
        self.kin_labels = CollectPreds("kin_labels")

        self.sample_cls = Sample

        print(self.hparams)

    def forward(self, inputs):
        return self.model(inputs)

    def _step(self, inputs):
        f1, f2, att = self(inputs)
        loss = self.loss_fn(f1, f2, beta=att)
        sim = torch.cosine_similarity(f1, f2)
        outputs = {"contrastive_loss": loss, "sim": sim, "features": [f1, f2, att]}
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, _ = batch
        loss = self._step([img1, img2])["contrastive_loss"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        outputs = self._step([img1, img2])
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)

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
                self.trainer.global_step > self.trainer.estimated_stepping_batches - self.hparams.cooldown
            ):  # cooldown start
                cur_lr = (self.trainer.estimated_stepping_batches - self.trainer.global_step) * (
                    optimizer.param_groups[0]["lr"] - self.hparams.end_lr
                ) / self.hparams.cooldown + self.hparams.end_lr
                optimizer.param_groups[0]["lr"] = cur_lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def on_train_epoch_end(self):
        # Calculate the number of samples processed
        use_sample = (
            (self.current_epoch + 1) * self.trainer.datamodule.batch_size * int(self.trainer.limit_train_batches)
        )
        # Update the dataset's bias or sampling strategy
        self.trainer.datamodule.train_dataset.set_bias(use_sample)

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")

    def _on_epoch_end(self, stage):
        # Compute predictions
        similarities = self.similarities.compute()
        is_kin_labels = self.is_kin_labels.compute()
        kin_labels = self.kin_labels.compute()
        self.__compute_metrics(similarities, is_kin_labels, kin_labels, stage=stage)
        # Reset predictions
        self.similarities.reset()
        self.is_kin_labels.reset()
        self.kin_labels.reset()

    def __compute_metrics(self, similarities, is_kin_labels, kin_labels, stage="train"):

        fpr, tpr, thresholds = tm.functional.roc(similarities, is_kin_labels, task="binary")
        maxindex = (tpr - fpr).argmax()

        # Compute best threshold
        if stage == "test" and self.threshold is None:
            raise ValueError("Threshold must be provided for test stage")
        elif stage == "test":
            best_threshold = self.threshold
        else:  # Compute best threshold for training or validation
            best_threshold = thresholds[maxindex].item()  # probability

        # Compute metrics
        #   - similarities will be converted to probabilites,
        #   - therefore best_threshold must be a probability
        if stage == "test":
            best_threshold = torch.sigmoid(torch.tensor(best_threshold)).item()
            # val stage computes its own threshold, which is already a probability

        auc = tm.functional.auroc(similarities, is_kin_labels, task="binary")
        acc = tm.functional.accuracy(similarities, is_kin_labels, threshold=best_threshold, task="binary")
        precision = tm.functional.precision(similarities, is_kin_labels, threshold=best_threshold, task="binary")
        recall = tm.functional.recall(similarities, is_kin_labels, threshold=best_threshold, task="binary")

        # Compute and log accuracy for each kinship relation
        #   -> best_threshold is a probability
        self.__compute_metrics_kin(similarities, is_kin_labels, kin_labels, best_threshold)
        self.__log_similarities(similarities, is_kin_labels)

        # Plot ROC curve and histogram of similarities (logits)
        best_threshold_logit = torch.logit(torch.tensor(best_threshold))
        self.__plot_roc_curve(auc, fpr, tpr, maxindex, similarities, is_kin_labels, best_threshold_logit)

        # Log metrics
        self.log("threshold", best_threshold_logit, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def __log_similarities(self, similarities, is_kin_labels):
        # Log similarities histogram by is_kin_labels
        positive = similarities[is_kin_labels == 1]
        negative = similarities[is_kin_labels == 0]
        if positive.numel() > 0:
            self.logger.experiment.add_histogram(
                "similarities/positive",
                positive,
                global_step=self.current_epoch,
            )
        if negative.numel() > 0:
            self.logger.experiment.add_histogram(
                "similarities/negative",
                negative,
                global_step=self.current_epoch,
            )

    def __compute_metrics_kin(self, similarities, is_kin_labels, kin_labels, best_threshold):
        for kin, kin_id in self.sample_cls.NAME2LABEL.items():  # TODO: pass Sample class as argument
            # TODO: fix non-kin accuracy compute
            mask = kin_labels == kin_id
            if torch.any(mask):
                acc = tm.functional.accuracy(
                    similarities[mask], is_kin_labels[mask].int(), threshold=best_threshold, task="binary"
                )
                self.log(
                    f"accuracy/{kin}",
                    acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
                # Add similarities
                # Negative pairs are "non-kin" pairs, which are equal to the overall similarities/negative
                positives = similarities[mask][is_kin_labels[mask] == 1]
                negatives = similarities[mask][is_kin_labels[mask] == 0]
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

    def __plot_roc_curve(self, auc, fpr, tpr, maxindex, similarities, is_kin_labels, best_threshold):
        # Convert to numpy
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        similarities = similarities.cpu().numpy()
        is_kin_labels = is_kin_labels.cpu().numpy()
        best_threshold = best_threshold.cpu().numpy()

        # Plot ROC Curve
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % auc)
        axs[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axs[0].scatter(fpr[maxindex], tpr[maxindex], s=50, c="red", label=f"Threshold ({best_threshold:.6f})")
        axs[0].set_xlim([0.0, 1.0])
        axs[0].set_ylim([0.0, 1.05])
        axs[0].set_xlabel("False Positive Rate")
        axs[0].set_ylabel("True Positive Rate")
        axs[0].set_title("Receiver Operating Characteristic")
        axs[0].legend(loc="lower right")

        # Plot Histogram of Similarities
        positives = [similarities[i] for i in range(len(similarities)) if is_kin_labels[i] == 1]
        negatives = [similarities[i] for i in range(len(similarities)) if is_kin_labels[i] == 0]

        axs[1].hist(positives, bins=20, alpha=0.5, label="Positive", color="g")
        axs[1].hist(negatives, bins=20, alpha=0.5, label="Negative", color="r")
        axs[1].axvline(x=best_threshold, color="b", linestyle="--", label=f"Threshold ({best_threshold:.6f})")
        axs[1].set_xlabel("Similarity")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Histogram of Similarities")
        axs[1].legend(loc="upper right")

        plt.tight_layout()

        self.logger.experiment.add_figure(
            "ROC Curve and Histogram of Similarities", fig, global_step=self.current_epoch
        )
        plt.close(fig)


# Define a custom L2 normalization layer
class L2Norm(nn.Module):
    def __init__(self, axis=1):
        super(L2Norm, self).__init__()
        self.axis = axis

    def forward(self, x):
        # L2 normalization
        return nn.functional.normalize(x, p=2, dim=self.axis)


class FaCoRNetMTFamily(FaCoRNetLightning):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nf = self.hparams.num_families
        self.classifier = HeadFamily(in_features=512, out_features=self.nf)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy_family = tm.Accuracy(
            num_classes=self.nf, task="multiclass"
        )  # nf will change for val and test, therefore won't be compute

    def _get_logits(self, x):
        if isinstance(self.loss, ArcFaceLoss):
            logits = self.loss.get_logits(x)
        else:
            logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):
        pair_batch, family_batch = batch
        img1, img2, labels = pair_batch
        imgs, families, _ = family_batch
        # Forward pass
        outputs = super()._step([img1, img2])
        fam_features, _ = self.model.backbone(imgs[:, [2, 1, 0]])  # why? the original code has this.
        fam_features = l2_norm(fam_features)
        fam_logits = self._get_logits(fam_features)
        # Compute losses
        contrastive_loss = outputs["contrastive_loss"]
        family_loss = self.loss(fam_logits, families)
        if self.hparams.loss_factor:
            loss = (1 - self.hparams.loss_factor) * contrastive_loss + self.hparams.loss_factor * family_loss
        else:
            loss = contrastive_loss + family_loss
        # Compute and log family accuracy
        family_accuracy = self.accuracy_family(fam_logits, families)
        self.log(
            "accuracy/classification/family", family_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        # Log lr and losses
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/contrastive/train", contrastive_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/classification/train", family_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        # Forward pass
        outputs = super()._step([img1, img2])
        # Compute losses
        contrastive_loss = outputs["contrastive_loss"]
        # Log losses
        self.log(f"loss/{stage}", contrastive_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FamilyClassifier(FaCoRNetLightning):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = load_pretrained_model("ir_101")
        self.nf = self.hparams.num_families
        self.classifier = HeadFamily(in_features=512, out_features=self.nf)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy_family = tm.Accuracy(
            num_classes=self.nf, task="multiclass"
        )  # nf will change for val and test, therefore won't be compute

    def forward(self, inputs):
        return self.model(inputs[:, [2, 1, 0]])[0]

    def training_step(self, batch, batch_idx):
        imgs, families, _ = batch
        # Forward pass
        fam_features = self(imgs)
        fam_features = l2_norm(fam_features)
        fam_preds = self.classifier(fam_features)
        # Compute losses
        loss = self.cross_entropy(fam_preds, families)
        # Compute and log family accuracy
        family_accuracy = self.accuracy_family(fam_preds, families)
        self.log(
            "accuracy/classification/family", family_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # Log lr and losses
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin = labels
        # Forward pass
        f1 = self(img1)
        f2 = self(img2)
        sim = torch.cosine_similarity(f1, f2)
        # Compute losses
        loss = contrastive_loss(f1, f2, beta=0.08)  # R2021
        # Log losses
        self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(sim)
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)


class FaCoRNetKinRace(FaCoRNetLightning):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_cls = SampleKFC

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin, _ = labels
        outputs = self._step([img1, img2])
        self.log(f"loss/{stage}", outputs["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)
