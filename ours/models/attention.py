import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import l2_norm

# FaCoRNet


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
            x : input feature maps( b x c x w x h)
        returns :
            out : self attention value + input feature
            attention: b x n x n (n is width*height)
        """
        m_batchsize, c, width, height = x1.size()
        x = torch.cat([x1, x2], 1)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # b x cx(n)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # b x c x (*w*h)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # bx (n) x (n)
        proj_value1 = self.value_conv1(x1)
        proj_value2 = self.value_conv2(x2)

        proj_value1 = proj_value1.view(m_batchsize, -1, width * height)  # b x c x n
        proj_value2 = proj_value2.view(m_batchsize, -1, width * height)  # b x c x n

        out1 = torch.bmm(proj_value1, attention.permute(0, 2, 1))
        out2 = torch.bmm(proj_value2, attention.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, -1, width, height) + x1.view(m_batchsize, -1, width, height)
        out2 = out2.view(m_batchsize, -1, width, height) + x2.view(m_batchsize, -1, width, height)
        # out = self.gamma*out + x.view(m_batchsize,2*c,width,height)
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
        # This is wrong: it inverts the attention
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # attention = self.softmax(energy_new)
        # This is correct
        attention = self.softmax(energy)

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


class FaCoRAttention(nn.Module):

    def __init__(self):
        super(FaCoRAttention, self).__init__()
        self.channel = 64
        self.spatial_ca = SpatialCrossAttention(self.channel * 8, CA=True)
        self.channel_ca = ChannelCrossAttention(self.channel * 8)
        self.channel_interaction = ChannelInteraction(1024)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, f1_0, x1_feat, f2_0, x2_feat):
        # (B, 49, 49)
        _, _, att_map0 = self.spatial_ca(x1_feat, x2_feat)

        # Both are (B, 512, 7, 7)
        x1_feat = l2_norm(x1_feat)
        x2_feat = l2_norm(x2_feat)

        # (B, 512)
        f1_1, f2_1, _ = self.channel_ca(f1_0, f2_0)
        # 2x (B, 512, 7, 7)
        f1_2, f2_2, _ = self.spatial_ca(x1_feat, x2_feat)

        # Both are (B, 512)
        f1_2 = torch.flatten(self.avg_pool(f1_2), 1)
        f2_2 = torch.flatten(self.avg_pool(f2_2), 1)

        # (B, 1024, 1, 1)
        wC = self.channel_interaction(torch.cat([f1_1, f1_2], 1).unsqueeze(2).unsqueeze(3))
        # (B, 2, 512, 1, 1)
        wC = wC.view(-1, 2, 512)[:, :, :, None, None]
        # (B, 512, 1, 1)
        f1s = f1_1.unsqueeze(2).unsqueeze(3) * wC[:, 0] + f1_2.unsqueeze(2).unsqueeze(3) * wC[:, 1]

        # (B, 1024, 1, 1)
        wC2 = self.channel_interaction(torch.cat([f2_1, f2_2], 1).unsqueeze(2).unsqueeze(3))
        # (B, 2, 512, 1, 1)
        wC2 = wC2.view(-1, 2, 512)[:, :, :, None, None]
        # (B, 512, 1, 1)
        f2s = f2_1.unsqueeze(2).unsqueeze(3) * wC2[:, 0] + f2_2.unsqueeze(2).unsqueeze(3) * wC2[:, 1]

        # (B, 512)
        f1s = torch.flatten(f1s, 1)
        f2s = torch.flatten(f2s, 1)

        return f1s, f2s, att_map0


class FaCoRAttentionV2(FaCoRAttention):

    def forward(self, f1_0, x1_feat, f2_0, x2_feat):

        # Both are (B, 512, 7, 7)
        x1_feat = l2_norm(x1_feat)
        x2_feat = l2_norm(x2_feat)

        # (B, 512)
        f1_1, f2_1, _ = self.channel_ca(f1_0, f2_0)
        # 2x (B, 512, 7, 7), (B, 49, 49)?
        f1_2, f2_2, att_map0 = self.spatial_ca(x1_feat, x2_feat)

        # Both are (B, 512)
        f1_2 = torch.flatten(self.avg_pool(f1_2), 1)
        f2_2 = torch.flatten(self.avg_pool(f2_2), 1)

        # (B, 1024, 1, 1)
        wC = self.channel_interaction(torch.cat([f1_1, f1_2], 1).unsqueeze(2).unsqueeze(3))
        # (B, 2, 512, 1, 1)
        wC = wC.view(-1, 2, 512)[:, :, :, None, None]
        # (B, 512, 1, 1)
        f1s = f1_1.unsqueeze(2).unsqueeze(3) * wC[:, 0] + f1_2.unsqueeze(2).unsqueeze(3) * wC[:, 1]

        # (B, 1024, 1, 1)
        wC2 = self.channel_interaction(torch.cat([f2_1, f2_2], 1).unsqueeze(2).unsqueeze(3))
        # (B, 2, 512, 1, 1)
        wC2 = wC2.view(-1, 2, 512)[:, :, :, None, None]
        # (B, 512, 1, 1)
        f2s = f2_1.unsqueeze(2).unsqueeze(3) * wC2[:, 0] + f2_2.unsqueeze(2).unsqueeze(3) * wC2[:, 1]

        # (B, 512)
        f1s = torch.flatten(f1s, 1)
        f2s = torch.flatten(f2s, 1)

        return f1s, f2s, att_map0


class FaCoRAttentionV3(FaCoRAttention):

    def forward(self, f1_0, x1_feat, f2_0, x2_feat):

        # Both are (B, 512, 7, 7)
        x1_feat = l2_norm(x1_feat)
        x2_feat = l2_norm(x2_feat)

        # 2x (B, 512), (B, 512, 512)
        f1_1, f2_1, att_map0 = self.channel_ca(f1_0, f2_0)
        # 2x (B, 512, 7, 7), (B, 49, 49)
        f1_2, f2_2, att_map1 = self.spatial_ca(x1_feat, x2_feat)

        # Both are (B, 512)
        f1_2 = torch.flatten(self.avg_pool(f1_2), 1)
        f2_2 = torch.flatten(self.avg_pool(f2_2), 1)

        # (B, 1024, 1, 1)
        wC = self.channel_interaction(torch.cat([f1_1, f1_2], 1).unsqueeze(2).unsqueeze(3))
        # (B, 2, 512, 1, 1)
        wC = wC.view(-1, 2, 512)[:, :, :, None, None]
        # (B, 512, 1, 1)
        f1s = f1_1.unsqueeze(2).unsqueeze(3) * wC[:, 0] + f1_2.unsqueeze(2).unsqueeze(3) * wC[:, 1]

        # (B, 1024, 1, 1)
        wC2 = self.channel_interaction(torch.cat([f2_1, f2_2], 1).unsqueeze(2).unsqueeze(3))
        # (B, 2, 512, 1, 1)
        wC2 = wC2.view(-1, 2, 512)[:, :, :, None, None]
        # (B, 512, 1, 1)
        f2s = f2_1.unsqueeze(2).unsqueeze(3) * wC2[:, 0] + f2_2.unsqueeze(2).unsqueeze(3) * wC2[:, 1]

        # (B, 512)
        f1s = torch.flatten(f1s, 1)
        f2s = torch.flatten(f2s, 1)

        return f1s, f2s, [att_map0, att_map1]


class FaCoRAttentionDummy(nn.Module):
    """
    Dummy FaCoR Attention module that does not perform any computation.

    Designed for FaCoRV5.
    """

    def __init__(self):
        super(FaCoRAttentionDummy, self).__init__()

    def forward(self, f1_0, x1_feat, f2_0, x2_feat):
        return f1_0, f2_0


# KFC


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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


class CBAM(torch.nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelGate(gate_channels)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_out = self.channel_gate(x)
        x_out = self.spatial_gate(x_out)
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


class KFCAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extract_flatten_feature = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 7x7x512 -> 1x1x512
            nn.Conv2d(in_channels=512, kernel_size=(1, 1), stride=(1, 1), out_channels=512),  # 1x1x512 -> 1x1x512
            nn.ReLU(),
        )
        self.cbam = CBAM(512)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512 * 7 * 7 + 512, kernel_size=(1, 1), stride=(1, 1), out_channels=1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, kernel_size=(1, 1), stride=(1, 1), out_channels=512),
            nn.ReLU(),
        )

    def extract_and_unsqueeze_features(self, feature, unsqueeze_dim):
        extracted_feature = self.extract_flatten_feature(feature).squeeze()
        return torch.unsqueeze(extracted_feature, unsqueeze_dim)

    def compute_correlation_map(self, F1, F2):
        return torch.exp(torch.matmul(F1, F2) / (512**0.5))

    def compute_cross_attention(self, correlation_map, compact_feature, dimension):
        denominator = torch.sum(correlation_map, dim=dimension)
        epsilon = 1e-5
        reshaped_compact = compact_feature.view(-1, 512, 7 * 7)
        cross_attention = torch.matmul(
            correlation_map if dimension == 2 else torch.permute(correlation_map, (0, 2, 1)), reshaped_compact
        )
        return cross_attention / (torch.unsqueeze(denominator, dim=2) + epsilon)

    def concatenate_features(self, embedding, aggregate_feature):
        concatenated = torch.cat([embedding, aggregate_feature], dim=1)
        return concatenated[..., None, None]

    def forward(self, embedding1, compact_feature1, embedding2, compact_feature2):
        # (B, 512, 1)
        F1 = self.extract_and_unsqueeze_features(compact_feature1, 2)
        # (B, 1, 512)
        F2 = self.extract_and_unsqueeze_features(compact_feature2, 1)

        # (B, 512, 512)
        correlation_map = self.compute_correlation_map(F1, F2)

        # (B, 512, 49)
        cross_attention_feature1 = self.compute_cross_attention(correlation_map, compact_feature1, 2)
        # (B, 512, 49)
        cross_attention_feature2 = self.compute_cross_attention(correlation_map, compact_feature2, 1)

        # (B, 25088)
        aggregate_feature1 = self.cbam(cross_attention_feature1.view(-1, 512, 7, 7) + compact_feature1)
        # (B, 25088)
        aggregate_feature2 = self.cbam(cross_attention_feature2.view(-1, 512, 7, 7) + compact_feature2)

        # (B, 25600, 1, 1) because of 512*7*7+512
        concate_feature1 = self.concatenate_features(embedding1, aggregate_feature1)
        concate_feature2 = self.concatenate_features(embedding2, aggregate_feature2)

        # (B, 512)
        x_out1 = torch.squeeze(self.conv(concate_feature1))
        x_out2 = torch.squeeze(self.conv(concate_feature2))

        # 2x (B, 25600), 2x (B, 512)
        return torch.squeeze(concate_feature1), torch.squeeze(concate_feature2), x_out1, x_out2


class KFCAttentionV2(KFCAttention):

    def __init__(self):
        super().__init__()
        self.conv_concat = nn.Sequential(
            nn.Conv2d(in_channels=512 * 2, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, embedding1, compact_feature1, embedding2, compact_feature2):
        concat_feature1, concat_feature2, x1, x2 = super().forward(
            embedding1, compact_feature1, embedding2, compact_feature2
        )
        # (B, 25600) -> (B, 2 * 512, 50)
        concat_feature = torch.cat([concat_feature1, concat_feature2], dim=1).view(-1, 2 * 512, 1, 1)
        # (B, 512, 50)
        concat_feature = self.conv_concat(concat_feature)
        return x1, x2, concat_feature.view(-1, 512, 50)


class KFCAttentionV3(KFCAttention):

    def forward(self, embedding1, compact_feature1, embedding2, compact_feature2):
        # (B, 512, 1)
        F1 = self.extract_and_unsqueeze_features(compact_feature1, 2)
        # (B, 1, 512)
        F2 = self.extract_and_unsqueeze_features(compact_feature2, 1)

        # (B, 512, 512)
        correlation_map = self.compute_correlation_map(F1, F2)

        # (B, 512, 49)
        cross_attention_feature1 = self.compute_cross_attention(correlation_map, compact_feature1, 2)
        # (B, 512, 49)
        cross_attention_feature2 = self.compute_cross_attention(correlation_map, compact_feature2, 1)

        # (B, 25088)
        aggregate_feature1 = self.cbam(cross_attention_feature1.view(-1, 512, 7, 7) + compact_feature1)
        # (B, 25088)
        aggregate_feature2 = self.cbam(cross_attention_feature2.view(-1, 512, 7, 7) + compact_feature2)

        # (B, 25600, 1, 1) because of 512*7*7+512
        concate_feature1 = self.concatenate_features(embedding1, aggregate_feature1)
        concate_feature2 = self.concatenate_features(embedding2, aggregate_feature2)

        # (B, 512)
        x_out1 = torch.squeeze(self.conv(concate_feature1))
        x_out2 = torch.squeeze(self.conv(concate_feature2))

        # (B, 512), (B, 512), (B, 512, 512)
        return x_out1, x_out2, correlation_map


if __name__ == "__main__":
    x1 = torch.randn(1, 512)
    x2 = torch.randn(1, 512)
    fmap1 = torch.randn(1, 512, 7, 7)
    fmap2 = torch.randn(1, 512, 7, 7)
    model = KFCAttention()
    model(x1, fmap1, x2, fmap2)
