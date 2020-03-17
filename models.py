import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ProjectorBlock, LinearAttentionBlock

"""
Implementation of ResNet
Reference: Deep Residual Learning for Image Recognition
"""
class BasicBlock(nn.Module):
    expansion = 1
    # planes refer to the number of feature maps
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        # conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # conv2
        out = self.conv2(out)
        out = self.bn2(out)
        # downsample
        if self.downsample is not None:
            residual = self.downsample(x)

        # print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    # planes refer to the number of feature maps
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # kernal_size=1 don't need padding
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # conv2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # conv3
        out = self.conv3(out)
        out = self.bn3(out)
        # downsample
        if self.downsample is not None:
            residual = self.downsample(x)

        # print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, attention=False, num_classes=10):
        super(ResNet, self).__init__()
        # initialize inplanes to 64, it'll be changed later
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layers refers to the number of blocks in each layer
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # attention blocks
        self.attention = attention
        if self.attention:
            self.attn1 = LinearAttentionBlock(in_channels=512*block.expansion, normalize_attn=True)
            self.attn2 = LinearAttentionBlock(in_channels=512*block.expansion, normalize_attn=True)
            self.attn3 = LinearAttentionBlock(in_channels=512*block.expansion, normalize_attn=True)
            self.attn4 = LinearAttentionBlock(in_channels=512*block.expansion, normalize_attn=True)
            self.projector1 = ProjectorBlock(in_channels=64*block.expansion, out_channels=512*block.expansion)
            self.projector2 = ProjectorBlock(in_channels=128*block.expansion, out_channels=512*block.expansion)
            self.projector3 = ProjectorBlock(in_channels=256*block.expansion, out_channels=512*block.expansion)
            self.fc = nn.Linear(512 * block.expansion * 4, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        # when the in-channel and the out-channel dismatch, downsample!!!
        if stride != 1 or self.inplanes != planes * block.expansion:
            # stride once for downsample and block.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        # only the first block needs downsample.
        layers.append(block(self.inplanes, planes, stride, downsample))
        # change inplanes for the next layer
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        g = self.avgpool(l4)
        # print(g.shape)
        # attention
        if self.attention:
            # print(l1.shape, l2.shape, l3.shape, l4.shape, g.shape)
            c1, g1 = self.attn1(self.projector1(l1), g)
            c2, g2 = self.attn2(self.projector2(l2), g)
            c3, g3 = self.attn3(self.projector3(l3), g)
            c4, g4 = self.attn4(l4, g)
            g = torch.cat((g1,g2,g3,g4), dim=1)
            x = self.fc(g)
        else:
            c1, c2, c3, c4 = None, None, None, None
            # x.size(0) ------ batch_size
            g = g.view(g.size(0), -1)
            x = self.fc(g)

        return [x, c1, c2, c3, c4]


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)

# Test
if __name__ == '__main__':
    model = ResNet18(attention=True, num_classes=10)
    x = torch.randn(1,3,128,128)
    print(model(x))
