import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ProjectorBlock, SpatialAttn, TemporalAttn
import math

"""
VGG-16 with attention
"""
class AttnVGG(nn.Module):
    def __init__(self, sample_size, num_classes, attention=True, normalize_attn=True, init_weights=True):
        super(AttnVGG, self).__init__()
        # conv blocks
        self.conv1 = self._make_layer(3, 64, 2)
        self.conv2 = self._make_layer(64, 128, 2)
        self.conv3 = self._make_layer(128, 256, 3)
        self.conv4 = self._make_layer(256, 512, 3)
        self.conv5 = self._make_layer(512, 512, 3)
        self.conv6 = self._make_layer(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(sample_size/32), padding=0, bias=True)
        # attention blocks
        self.attention = attention
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=512*3, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # if init_weights:
        #     self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        l1 = self.conv3(x)
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)
        l2 = self.conv4(x)
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)
        l3 = self.conv5(x)
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0)
        x = self.conv6(x)
        g = self.dense(x) # batch_sizex512x1x1
        # attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizex3C
            # classification layer
            x = self.classify(g) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return [x, c1, c2, c3]

    def _make_layer(self, in_features, out_features, blocks, pool=False):
        layers = []
        for i in range(blocks):
            conv2d = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1, bias=False)
            layers += [conv2d, nn.BatchNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features
            if pool:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

"""
LSTM with attention
"""
class AttnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(AttnLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        self.attn = TemporalAttn(hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x, weights = self.attn(x)
        x = self.fc(x)
        return x, weights

# Test
if __name__ == '__main__':
    model = AttnVGG(sample_size=128, num_classes=10)
    x = torch.randn(16,3,128,128)
    print(model(x))
    model = AttnLSTM(input_size=1, hidden_size=128, num_layers=1)
    x = torch.randn(16, 20, 1)
    print(model(x))
