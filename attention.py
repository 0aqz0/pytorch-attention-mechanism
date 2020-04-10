import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention blocks
Reference: Learn To Pay Attention
"""
# class ProjectorBlock(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(ProjectorBlock, self).__init__()
#         self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features,
#             kernel_size=1, padding=0, bias=False)

#     def forward(self, x):
#         return self.op(x)


# class LinearAttentionBlock(nn.Module):
#     def __init__(self, in_features, normalize_attn=True):
#         super(LinearAttentionBlock, self).__init__()
#         self.normalize_attn = normalize_attn
#         self.op = nn.Conv2d(in_channels=in_features, out_channels=1,
#             kernel_size=1, padding=0, bias=False)

#     def forward(self, l, g):
#         N, C, H, W = l.size()
#         c = self.op(l+g) # (batch_size,1,H,W)
#         if self.normalize_attn:
#             a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,H,W)
#         else:
#             a = torch.sigmoid(c)
#         g = torch.mul(a.expand_as(l), l)
#         if self.normalize_attn:
#             g = g.view(N,C,-1).sum(dim=2) # (batch_size,C)
#         else:
#             g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
#         return c.view(N,1,H,W), g

"""
Spatial & Channel Attention Blocks
Reference: BAM: Bottleneck Attention Module (BMVC2018)
"""
class ChannelAttn(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttn, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels//reduction_ratio)
        self.bn1 = nn.BatchNorm1d(in_channels//reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels//reduction_ratio, in_channels)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1,1))
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.relu(self.bn1(self.fc1(out)))
        out = self.fc2(out)
        # print(out.shape)
        return out.unsqueeze(2).unsqueeze(3).expand_as(x)

class SpatialAttn(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpatialAttn, self).__init__()
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels//reduction_ratio)
        self.relu = nn.ReLU()
        # 3x3 conv * 2
        self.conv2 = nn.Conv2d(in_channels//reduction_ratio, in_channels//reduction_ratio, kernel_size=3,
                                padding=4, dilation=4)
        self.bn2 = nn.BatchNorm2d(in_channels//reduction_ratio)
        self.conv3 = nn.Conv2d(in_channels//reduction_ratio, in_channels//reduction_ratio, kernel_size=3,
                                padding=4, dilation=4)
        self.bn3 = nn.BatchNorm2d(in_channels//reduction_ratio)
        # 1x1 conv
        self.conv4 = nn.Conv2d(in_channels//reduction_ratio, 1, kernel_size=1)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        return out.expand_as(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.channel_attn = ChannelAttn(in_channels)
        self.spatial_attn = SpatialAttn(in_channels)

    def forward(self, x):
        attn = 1 + torch.sigmoid(self.channel_attn(x) + self.spatial_attn(x))
        # print(attn.shape)
        return attn * x

# Test
if __name__ == '__main__':
    # # 2d block
    # attention_block = LinearAttentionBlock(in_features=3)
    # l = torch.randn(16, 3, 128, 128)
    # g = torch.randn(16, 3, 128, 128)
    # print(attention_block(l, g))
    # channel block
    channel_block = ChannelAttn(in_channels=128)
    x = torch.randn(16, 128, 8, 8)
    print(channel_block(x).shape)
    # spatial block
    spatial_block = SpatialAttn(in_channels=128)
    x = torch.randn(16, 128, 8, 8)
    print(spatial_block(x).shape)
    # attn block
    attn_block = AttentionBlock(in_channels=128)
    x = torch.randn(16, 128, 8, 8)
    print(attn_block(x).shape)
