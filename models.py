import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from attention import AttentionBlock
import math

"""
Implementation of ResNet
Reference: Deep Residual Learning for Image Recognition
"""
class BasicBlock(nn.Module):
    expansion = 1
    # planes refer to the number of feature maps
    def __init__(self, inplanes, planes, stride=1, downsample=None, attention=False):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if attention:
            self.attn = AttentionBlock(planes)
        else:
            self.attn = None

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
        # attention
        if self.attn is not None:
            out = self.attn(out)

        # print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    # planes refer to the number of feature maps
    def __init__(self, inplanes, planes, stride=1, downsample=None, attention=False):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # kernal_size=1 don't need padding
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        if attention:
            self.attn = AttentionBlock(planes * 4)
        else:
            self.attn = None

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
        # attention
        if self.attn is not None:
            out = self.attn(out)

        # print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, attention=False):
        super(ResNet, self).__init__()
        # initialize inplanes to 64, it'll be changed later
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layers refers to the number of blocks in each layer
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, attention=attention)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, attention=attention)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, attention=attention)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, attention=attention)
        # average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # attention blocks
        self.attention = attention
        if self.attention:
            self.attn1 = AttentionBlock(in_channels=64*block.expansion)
            self.attn2 = AttentionBlock(in_channels=128*block.expansion)
            self.attn3 = AttentionBlock(in_channels=256*block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride, attention):
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
        layers.append(block(self.inplanes, planes, stride, downsample, attention=attention))
        # change inplanes for the next layer
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention=attention))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.attention:
            x = self.attn1(x)
        x = self.layer2(x)
        if self.attention:
            x = self.attn2(x)
        x = self.layer3(x)
        if self.attention:
            x = self.attn3(x)
        x = self.layer4(x)

        g = self.avgpool(x)
        g = g.view(g.size(0), -1)
        x = self.fc(g)

        return x

    def load_my_state_dict(self, state_dict):
        my_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name == 'fc.weight' or name == 'fc.bias':
                continue
            my_state_dict[name].copy_(param.data)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def ResNet18(pretrained=False, progress=True, **kwargs):
    model = ResNet(BasicBlock, [2,2,2,2], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18'],
            progress=progress)
        model.load_my_state_dict(state_dict)
    return model

def ResNet34(pretrained=False, progress=True, **kwargs):
    model = ResNet(BasicBlock, [3,4,6,3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet34'],
            progress=progress)
        model.load_my_state_dict(state_dict)
    return model

def ResNet50(pretrained=False, progress=True, **kwargs):
    model = ResNet(Bottleneck, [3,4,6,3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
            progress=progress)
        model.load_my_state_dict(state_dict)
    return model

def ResNet101(pretrained=False, progress=True, **kwargs):
    model = ResNet(Bottleneck, [3,4,23,3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101'],
            progress=progress)
        model.load_my_state_dict(state_dict)
    return model

def ResNet152(pretrained=False, progress=True, **kwargs):
    model = ResNet(Bottleneck, [3,8,36,3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152'],
            progress=progress)
        model.load_my_state_dict(state_dict)
    return model

# Test
if __name__ == '__main__':
    model = ResNet18(attention=True, num_classes=10)
    x = torch.randn(16,3,128,128)
    print(model(x))
