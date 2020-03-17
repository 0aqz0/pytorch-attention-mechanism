import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention blocks
Reference: Learn To Pay Attention
"""
class ProjectorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.op(x)


class LinearAttentionBlock(nn.Module):
    def __init__(self, in_channels, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_channels, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, H, W = l.size()
        c = self.op(l+g) # (batch_size,1,H,W)
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,H,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # (batch_size,C)
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,H,W), g

# Test
if __name__ == '__main__':
    # 2d block
    attention_block = LinearAttentionBlock(in_channels=3)
    l = torch.randn(16, 3, 128, 128)
    g = torch.randn(16, 3, 128, 128)
    print(attention_block(l, g))
