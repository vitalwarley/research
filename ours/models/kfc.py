import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import SampleKFC
from models.base import CollectPreds, LightningBaseModel, load_pretrained_model
from torch.autograd import Function

from datasets.kfc import RACE_DICT


class ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)  # make x be same shape as x

    @staticmethod
    def backward(ctx, *grad_output):  # grad_output backward gradient
        # result , = ctx.saved_tensors # the saving forward gradient
        # print(grad_output)
        return -(grad_output[0]) * ctx.alpha, None  # sine input is tuple


def grad_reverse(x, alpha=1.0):
    return ReverseLayer.apply(x, alpha)


# ctx: is a pytorch syntex that can save the variable in function


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CBAM(torch.nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class ChannelGate(torch.nn.Module):
    def __init__(self, in_channel, ratio=16):
        super().__init__()
        self.atten_module = nn.Sequential(
            Flatten(), nn.Linear(in_channel, in_channel // 16), nn.ReLU(), nn.Linear(in_channel // 16, in_channel)
        )

    def forward(self, x):
        channel_att_sum = None
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_avg = self.atten_module(avg_pool)
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_max = self.atten_module(max_pool)
        channel_att_sum = channel_att_avg + channel_att_max
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x = x * scale
        return x


class ChannelPool(torch.nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        self.flatten = nn.Sequential(nn.Flatten())

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        x = x * scale
        return self.flatten(x)


class Atten(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extract_flatten_feature = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 7x7x512 -> 1x1x512
            nn.Conv2d(in_channels=512, kernel_size=(1, 1), stride=(1, 1), out_channels=512),  # 1x1x512 -> 1x1x512
            nn.ReLU(),
        )
        self.CBAM_module = CBAM(512)
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=512 * 7 * 7 + 512, kernel_size=(1, 1), stride=(1, 1), out_channels=1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, kernel_size=(1, 1), stride=(1, 1), out_channels=512),
            nn.ReLU(),
        )

    def forward(self, embedding1, compact_feature1, embedding2, compact_feature2):
        F1 = torch.unsqueeze(torch.squeeze(self.extract_flatten_feature(compact_feature1)), 2)  # 1x1x512 -> 512
        F2 = torch.unsqueeze(torch.squeeze(self.extract_flatten_feature(compact_feature2)), 1)  # 1x1x512 -> 512

        correlation_map = torch.exp(torch.matmul(F1, F2) / (512**0.5))  # (_,512,512)
        denominator1 = torch.sum(correlation_map, dim=2)  # sum of each col
        denominator2 = torch.sum(correlation_map, dim=1)  # sum of each row

        cross_attention_feature1 = torch.matmul(
            correlation_map, compact_feature1.view(-1, 512, 7 * 7)
        ) / torch.unsqueeze(
            denominator1, dim=2
        )  # (_,256,) broadcast to each col ->
        cross_attention_feature2 = torch.matmul(
            torch.permute(correlation_map, (0, 2, 1)), compact_feature2.view(-1, 512, 7 * 7)
        ) / torch.unsqueeze(
            denominator2, dim=2
        )  # (_,512,7,1) broadcast to each col

        aggregate_feature1 = self.CBAM_module(cross_attention_feature1.view(-1, 512, 7, 7) + compact_feature1)
        aggregate_feature2 = self.CBAM_module(cross_attention_feature2.view(-1, 512, 7, 7) + compact_feature2)

        concate_feature1 = (torch.cat([embedding1, aggregate_feature1], dim=1))[..., None, None]  # 512*7*7+512
        concate_feature2 = (torch.cat([embedding2, aggregate_feature2], dim=1))[..., None, None]  # 512*7*7+512

        x_out1 = torch.squeeze(self.Conv(concate_feature1))
        x_out2 = torch.squeeze(self.Conv(concate_feature2))

        return torch.squeeze(concate_feature1), torch.squeeze(concate_feature2), x_out1, x_out2


class HeadRace(nn.Module):
    def __init__(self, in_features=512, out_features=4, ratio=8):
        super().__init__()
        self.projection_head = nn.Sequential(
            torch.nn.Linear(in_features, in_features // ratio),
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


class Debias(nn.Module):
    def __init__(self, in_feature, out_feature=512):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Debias = nn.Linear(in_feature, out_feature)

    def forward(self, x1, x2):
        Blen = len(x1) * 2
        x1x2 = torch.cat([x1, x2], dim=0)
        x2x1 = torch.cat([x2, x1], dim=0)

        middle = ((torch.unsqueeze(x1x2, dim=1) + torch.unsqueeze(x1x2, dim=0)) / 2).view(
            -1, self.in_feature
        )  # add by broadcast
        fmiddle = self.Debias(middle).view(Blen, Blen, -1)

        fx1x2 = self.Debias(x1x2)
        fx2x1 = self.Debias(x2x1)

        bias_map = (
            torch.cosine_similarity(torch.unsqueeze(fx1x2, dim=1), fmiddle, dim=2) ** 2
            - torch.cosine_similarity(torch.unsqueeze(fx1x2, dim=0), fmiddle, dim=2) ** 2
        )

        pair_mid = (x1x2 + x2x1) / 2
        fmix = self.Debias(pair_mid)
        bias_pair = torch.cosine_similarity(fx1x2, fmix, dim=1) ** 2 - torch.cosine_similarity(fx2x1, fmix, dim=1) ** 2

        return bias_map, bias_pair


class KFC(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder=KitModel("./kit_resnet101.pkl")
        self.encoder = load_pretrained_model("ir_101")
        self.Atten_module = Atten()
        self.task_race = HeadRace(512, 4, 8)
        self.debias_layer = Debias(512 * 7 * 7 + 512)

    def forward(self, imgs):
        img1, img2 = imgs
        idx = [2, 1, 0]
        e1, compact_feature1 = self.encoder(img1[:, idx])  # each featur 16, 512x7x7
        e2, compact_feature2 = self.encoder(img2[:, idx])
        atten_em1, atten_em2, x1, x2 = self.Atten_module(e1, compact_feature1, e2, compact_feature2)
        reverse_em1, reverse_em2 = grad_reverse(e1, 1.0), grad_reverse(e2, 1.0)
        r1, r2 = self.task_race(reverse_em1), self.task_race(reverse_em2)
        # r1,r2=self.task_race(e1),self.task_race(e2)
        biasmap, bias_pair = self.debias_layer(atten_em1, atten_em2)

        return r1, r2, e1, e2, x1, x2, biasmap, bias_pair

    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), path)

    def load(self, path, num=0):
        self.encoder.load_state_dict(torch.load(os.path.join(path, f"encoder{num}.pth")))
        self.Atten_module.load_state_dict(torch.load(os.path.join(path, f"Atten{num}.pth")))
        self.task_race.load_state_dict(torch.load(os.path.join(path, f"task_race{num}.pth")))
        self.debias_layer.load_state_dict(torch.load(os.path.join(path, f"debias_layer{num}.pth")))

    def save(self, path, num=0):
        self.save_model(self.encoder, os.path.join(path, f"encoder{num}.pth"))
        self.save_model(self.Atten_module, os.path.join(path, f"Atten{num}.pth"))
        self.save_model(self.task_race, os.path.join(path, f"task_race{num}.pth"))
        self.save_model(self.debias_layer, os.path.join(path, f"debias_layer{num}.pth"))


class KFCLightning(LightningBaseModel):

    def __init__(self, **kwargs):
        super(KFCLightning, self).__init__(**kwargs)
        self.race_labels = CollectPreds("race_labels")
        self.sample_cls = SampleKFC
        print(self.hparams)

        # Define the mapping of indices to race keys, simplify to first part before '&'
        self.possible_races = [k.split("&")[0] for k, v in sorted(RACE_DICT.items(), key=lambda item: item[1][0])]

    def _step(self, inputs, labels):
        pred_races1, pred_races2, e1, e2, f1, f2, bias_map, bias_pair = self(inputs)

        c_loss, margins = self.criterion[0](f1, f2, labels, bias_map)
        sim = torch.cosine_similarity(e1, e2)  # backbone features instead of attention features
        outputs = {
            "contrastive_loss": c_loss,
            "margins": margins,
            "sim": sim,
            "predictions": [pred_races1, pred_races2],
            "features": [f1, f2],
        }
        return outputs

    def training_step(self, batch, batch_idx):
        img1, img2, labels = batch
        outputs = self._step([img1, img2], labels[-1])
        c_loss = outputs["contrastive_loss"]
        pred_races1, pred_races2 = outputs["predictions"]
        race_loss = self.criterion[1](pred_races1, labels[-1]) + self.criterion[1](pred_races2, labels[-1])
        total_loss = c_loss + race_loss
        margins = outputs["margins"]
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # on_step=True to see the warmup and cooldown properly :)
        self.log("lr", cur_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/train", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/contrastive", c_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss/train/race", race_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Log margins for AA, A, C, I
        for race, margin in zip(["AA", "A", "C", "I"], margins):
            self.log(f"margin/train/{race}", margin, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return total_loss

    def _eval_step(self, batch, batch_idx, stage):
        img1, img2, labels = batch
        kin_relation, is_kin, race = labels
        outputs = self._step([img1, img2], race)
        c_loss = outputs["contrastive_loss"]
        margins = outputs["margins"]
        self.log(f"loss/{stage}/contrastive", c_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Log margins for AA, A, C, I
        for race_name, margin in zip(["AA", "A", "C", "I"], margins):
            self.log(f"margin/{stage}/{race_name}", margin, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # Compute best threshold for training or validation
        self.similarities(outputs["sim"])
        self.is_kin_labels(is_kin)
        self.kin_labels(kin_relation)
        self.race_labels(race)

    def _compute_race_metrics(self, stage):

        race_labels = self.race_labels.compute()

        # Convert race_label to simplified race keys using the prepared list
        race_names = [self.possible_races[idx.item()] for idx in race_labels]  # race name for each sample

        # Create a tensor of unique races to index
        unique_races = ["A", "AA", "C", "I"]
        race_index = {race: i for i, race in enumerate(unique_races)}
        race_indices_tensor = torch.tensor(
            [race_index[race] for race in race_names], dtype=torch.int, device=self.device
        )  # race id for each sample

        # Initialize count tensors
        race_table_tensor = torch.zeros(len(unique_races), dtype=torch.int, device=self.device)
        race_total_tensor = torch.zeros(len(unique_races), dtype=torch.int, device=self.device)

        # Count matches where prediction is true
        race_table_tensor.index_add_(
            0, race_indices_tensor[self.y_pred], torch.ones_like(race_indices_tensor[self.y_pred])
        )

        # Count total occurrences
        race_total_tensor.index_add_(0, race_indices_tensor, torch.ones_like(race_indices_tensor))

        # Convert the count tensors back to Python dictionaries
        race_table = {race: race_table_tensor[i].item() for i, race in enumerate(unique_races)}
        race_total = {race: race_total_tensor[i].item() for i, race in enumerate(unique_races)}

        # Log accuracy for each race
        accs = []
        for race in unique_races:
            if race_total[race]:
                race_acc = race_table[race] / race_total[race]
            else:
                race_acc = 0
            accs.append(race_acc)
            self.log(
                f"accuracy/{stage}/race/{race}", race_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True
            )

        # Log mean and std accuracy across races
        accs = torch.tensor(accs)
        mean_acc = torch.mean(accs)
        std_acc = torch.std(accs)

        # Log std and mean accuracy
        self.log(f"accuracy/{stage}/race/mean", mean_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"accuracy/{stage}/race/std", std_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def _on_epoch_end(self, stage):
        super(KFCLightning, self)._on_epoch_end(stage)
        self._compute_race_metrics(stage)
        self.race_labels.reset()
