import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention blocks
Reference: Learn To Pay Attention
"""
class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features,
            kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.op(x)


class SpatialAttn(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(SpatialAttn, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1,
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

"""
Temporal attention block
Reference: https://github.com/philipperemy/keras-attention-mechanism
"""
class TemporalAttn(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttn, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        # (batch_size, time_steps, hidden_size)
        score_first_part = self.fc1(hidden_states)
        # (batch_size, hidden_size)
        h_t = hidden_states[:,-1,:]
        # (batch_size, time_steps)
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)
        # (batch_size, hidden_size)
        context_vector = torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)
        # (batch_size, hidden_size*2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        # (batch_size, hidden_size)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector, attention_weights

# Test
if __name__ == '__main__':
    # 2d block
    spatial_block = SpatialAttn(in_features=3)
    l = torch.randn(16, 3, 128, 128)
    g = torch.randn(16, 3, 128, 128)
    print(spatial_block(l, g))
    # temporal block
    temporal_block = TemporalAttn(hidden_size=256)
    x = torch.randn(16, 30, 256)
    print(temporal_block(x).shape)
