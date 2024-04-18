import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, embedding1, compact_feature1, embedding2, compact_feature2):
        # (B, 512)
        extract_feature1 = self.extract_flatten_feature(compact_feature1).squeeze()
        extract_feature2 = self.extract_flatten_feature(compact_feature2).squeeze()

        # (B, 512, 1)
        F1 = torch.unsqueeze(extract_feature1, 2)
        # (B, 1, 512)
        F2 = torch.unsqueeze(extract_feature2, 1)

        # (B, 512, 512)
        correlation_map = torch.exp(torch.matmul(F1, F2) / (512**0.5))  # (_,512,512)
        # (B, 512)
        denominator1 = torch.sum(correlation_map, dim=2)  # sum of each col
        # (B, 512)
        denominator2 = torch.sum(correlation_map, dim=1)  # sum of each row

        # (B, 512, 49)
        epsilon = 1e-5
        cross_attention_feature1 = torch.matmul(correlation_map, compact_feature1.view(-1, 512, 7 * 7)) / (
            torch.unsqueeze(denominator1, dim=2) + epsilon
        )  # (_,256,) broadcast to each col ->
        # (B, 512, 49)
        cross_attention_feature2 = torch.matmul(
            torch.permute(correlation_map, (0, 2, 1)), compact_feature2.view(-1, 512, 7 * 7)
        ) / (
            torch.unsqueeze(denominator2, dim=2) + epsilon
        )  # (_,512,7,1) broadcast to each col

        # (B, 25088)
        aggregate_feature1 = self.cbam(cross_attention_feature1.view(-1, 512, 7, 7) + compact_feature1)
        aggregate_feature2 = self.cbam(cross_attention_feature2.view(-1, 512, 7, 7) + compact_feature2)

        # (B, 25600, 1, 1)
        concate_feature1 = (torch.cat([embedding1, aggregate_feature1], dim=1))[..., None, None]  # 512*7*7+512
        concate_feature2 = (torch.cat([embedding2, aggregate_feature2], dim=1))[..., None, None]  # 512*7*7+512

        # (B, 512)
        x_out1 = torch.squeeze(self.conv(concate_feature1))
        x_out2 = torch.squeeze(self.conv(concate_feature2))

        # (B, 25600)
        concate_feature1 = torch.squeeze(concate_feature1)
        concate_feature2 = torch.squeeze(concate_feature2)

        return concate_feature1, concate_feature2, x_out1, x_out2


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


if __name__ == "__main__":
    x1 = torch.randn(1, 512)
    x2 = torch.randn(1, 512)
    fmap1 = torch.randn(1, 512, 7, 7)
    fmap2 = torch.randn(1, 512, 7, 7)
    model = KFCAttention()
    model(x1, fmap1, x2, fmap2)
