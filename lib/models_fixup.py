# Code adapted from: https://github.com/aleximmer/ntk-marglik/blob/main/ntkmarglik/models.py
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def get_activation(act_str):
    if act_str == 'relu':
        return nn.ReLU
    elif act_str == 'tanh':
        return nn.Tanh
    elif act_str == 'selu':
        return nn.SELU
    elif act_str == 'silu':
        return nn.SiLU
    else:
        raise ValueError('invalid activation')


class Bias(nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0)

    def forward(self, input):
        return input + self.weight


class Scale(nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.weight = nn.Parameter(torch.ones(1))
        
    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)
        
    def forward(self, input):
        return self.weight * input


class MLP(nn.Sequential):
    def __init__(self, input_size, width, depth, output_size, activation='relu',
                 bias=True, fixup=False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        flatten_start_dim = 1
        act = get_activation(activation)
        output_size = output_size

        self.add_module('flatten', nn.Flatten(start_dim=flatten_start_dim))

        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, output_size, bias=bias))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', nn.Linear(in_size, out_size, bias=bias))
                if fixup:
                    self.add_module(f'bias{i+1}b', Bias())
                    self.add_module(f'scale{i+1}b', Scale())
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', nn.Linear(hidden_sizes[-1], output_size, bias=bias))


class WRNFixupBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, fixup=True):
        super(WRNFixupBasicBlock, self).__init__()
        self.bias1 = Bias() if fixup else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        basemodul = nn.Conv2d
        self.conv1 = basemodul(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bias2 = Bias() if fixup else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        self.bias3 = Bias() if fixup else nn.Identity()
        self.conv2 = basemodul(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias4 = Bias() if fixup else nn.Identity()
        self.scale1 = Scale() if fixup else nn.Identity()
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and basemodul(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bias1(x))
        else:
            out = self.relu1(self.bias1(x))
        if self.equalInOut:
            out = self.bias3(self.relu2(self.bias2(self.conv1(out))))
        else:
            out = self.bias3(self.relu2(self.bias2(self.conv1(x))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.bias4(self.scale1(self.conv2(out)))
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class WRNFixupNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, fixup=True):
        super(WRNFixupNetworkBlock, self).__init__()
        self.fixup = fixup
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, self.fixup))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetBase(nn.Module):
    def __init__(self, depth=16, n_channels=3, widen_factor=4, dropRate=0.0, num_classes=10, fixup=True):
        super(WideResNetBase, self).__init__()
        n_out = num_classes
        self.fixup = fixup
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = WRNFixupBasicBlock
        # 1st conv before any network block
        self.num_layers = n * 3
        basemodul = nn.Conv2d
        self.conv1 = basemodul(n_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias1 = Bias() if fixup else nn.Identity()
        # 1st block
        self.block1 = WRNFixupNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, fixup=fixup)
        # 2nd block
        self.block2 = WRNFixupNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, fixup=fixup)
        # 3rd block
        self.block3 = WRNFixupNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, fixup=fixup)
        # global average pooling and classifier
        self.bias2 = Bias() if fixup else nn.Identity()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(nChannels[3], n_out)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, WRNFixupBasicBlock):
                conv = m.conv1
                k = conv.weight.shape[0] * np.prod(conv.weight.shape[2:])
                nn.init.normal_(conv.weight,
                                mean=0,
                                std=np.sqrt(2. / k) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.convShortcut is not None:
                    cs = m.convShortcut
                    k = cs.weight.shape[0] * np.prod(cs.weight.shape[2:])
                    nn.init.normal_(cs.weight,
                                    mean=0,
                                    std=np.sqrt(2. / k))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.bias1(self.conv1(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.flatten(start_dim=1)
        out = self.bias2(out)
        # out = self.fc(out)
        return out
