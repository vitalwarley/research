import torch
import torch.nn as nn


def l2_norm(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    return x / norm


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
