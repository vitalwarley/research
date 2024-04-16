import numpy as np
import torch
import torch.nn as nn
import torchmetrics as tm
from datasets.utils import SampleKFC
from losses import contrastive_loss
from models.base import LightningBaseModel, load_pretrained_model
from pytorch_metric_learning.losses import ArcFaceLoss


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


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


# Define a custom L2 normalization layer
class L2Norm(nn.Module):
    def __init__(self, axis=1):
        super(L2Norm, self).__init__()
        self.axis = axis

    def forward(self, x):
        # L2 normalization
        return nn.functional.normalize(x, p=2, dim=self.axis)


class FaCoRNetLightning(LightningBaseModel):

    def _step(self, inputs):
        f1, f2, att = self(inputs)
        loss = self.criterion(f1, f2, beta=att)
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
