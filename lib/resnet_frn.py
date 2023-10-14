import torch 
import torch.nn as nn
from typing import Tuple

# # From: https://raw.githubusercontent.com/izmailovpavel/neurips_bdl_starter_kit/main/pytorch_models.py
class FilterResponseNorm_layer(nn.Module):
    def __init__(self, num_filters, eps=1e-6):
        super(FilterResponseNorm_layer, self).__init__()
        self.eps = eps
        par_shape = (1, num_filters, 1, 1)  # [1,C,1,1]
        self.tau = torch.nn.Parameter(torch.zeros(par_shape))
        self.beta = torch.nn.Parameter(
            torch.zeros(par_shape))
        self.gamma = torch.nn.Parameter(
            torch.ones(par_shape))

    def forward(self, x):
        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * 1 / torch.sqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        z = torch.max(y, self.tau)
        return z


# cf. https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
# should replicate tensorflow "SAME" padding behavior
def conv_same_padding(
        in_size: int, kernel: int, stride: int = 1, dilation: int = 1) -> Tuple[int, int]:
    effective_filter_size = (kernel - 1) * dilation + 1
    out_size = (in_size + stride - 1) // stride
    padding_needed = max(
        0, (out_size - 1) * stride + effective_filter_size - in_size)
    if padding_needed % 2 == 0:
        padding_left = padding_needed // 2
        padding_right = padding_needed // 2
    else:
        padding_left = (padding_needed - 1) // 2
        padding_right = (padding_needed + 1) // 2
    return padding_left, padding_right


class resnet_block(nn.Module):
    def __init__(
            self, normalization_layer, input_size, num_filters, kernel_size=3,
            strides=1, activation=torch.nn.Identity, use_bias=True):
        super(resnet_block, self).__init__()
        # input size = C, H, W
        p0, p1 = conv_same_padding(input_size[2], kernel_size, strides)
        # height padding
        p2, p3 = conv_same_padding(input_size[1], kernel_size, strides)
        self.pad1 = torch.nn.ZeroPad2d((p0, p1, p2, p3))
        self.conv1 = torch.nn.Conv2d(
            input_size[0], num_filters, kernel_size=kernel_size,
            stride=strides, padding=0, bias=use_bias)
        self.norm1 = normalization_layer(num_filters)
        self.activation1 = activation()

    def forward(self, x):

        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation1(out)

        return out


class stacked_resnet_block(nn.Module):
    def __init__(self, normalization_layer, num_filters, input_num_filters,
                 stack, res_block, activation, use_bias):
        super(stacked_resnet_block, self).__init__()
        self.stack = stack
        self.res_block = res_block
        spatial_out = 32 // (2 ** stack)
        if stack > 0 and res_block == 0:  # first layer but not first stack
            strides = 2  # downsample
        else:
            strides = 1
        spatial_in = spatial_out * strides

        self.res1 = resnet_block(
            normalization_layer=normalization_layer, num_filters=num_filters,
            input_size=(input_num_filters, spatial_in, spatial_in),
            strides=strides, activation=activation, use_bias=use_bias)
        self.res2 = resnet_block(
            normalization_layer=normalization_layer, num_filters=num_filters,
            input_size=(num_filters, spatial_out, spatial_out),
            use_bias=use_bias)
        if stack > 0 and res_block == 0:  # first layer but not first stack
            # linear projection residual shortcut to match changed dims
            self.res3 = resnet_block(
                normalization_layer=normalization_layer,
                num_filters=num_filters,
                input_size=(input_num_filters, spatial_in, spatial_in),
                strides=strides,
                kernel_size=1,
                use_bias=use_bias)

        self.activation1 = activation()

    def forward(self, x):

        y = self.res1(x)
        y = self.res2(y)
        if self.stack > 0 and self.res_block == 0:
            x = self.res3(x)
        out = self.activation1(x + y)
        return out


class make_resnet_fn(nn.Module):
    def __init__(self, num_classes, depth, normalization_layer,
                 width=16, use_bias=True, activation=torch.nn.ReLU(inplace=True)):
        super(make_resnet_fn, self).__init__()
        self.output_size = 10
        self.num_res_blocks = (depth - 2) // 6
        self.normalization_layer = normalization_layer
        self.activation = activation
        self.use_bias = use_bias
        self.width = width
        if (depth - 2) % 6 != 0:
            raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

        # first res_layer
        self.layer1 = resnet_block(normalization_layer=normalization_layer, num_filters=width,
                                   input_size=(3, 32, 32), kernel_size=3, strides=1,
                                   activation=torch.nn.Identity, use_bias=True)
        # stacks
        self.stacks = self._make_res_block()
        # avg pooling
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=(8, 8), stride=8, padding=0)
        # linear layer
        # self.linear1 = nn.Linear(64, num_classes)
        self.n_features = 64

    def forward(self, x):
        # first res_layer
        out = self.layer1(x)  # shape out torch.Size([5, 16, 32, 32])
        out = self.stacks(out)
        out = self.avgpool1(out)
        out = torch.flatten(out, start_dim=1)
        # logits = self.linear1(out)
        # return logits
        return out

    def _make_res_block(self):
        layers = list()
        num_filters = self.width
        input_num_filters = num_filters
        for stack in range(3):
            for res_block in range(self.num_res_blocks):
                layers.append(stacked_resnet_block(self.normalization_layer, num_filters, input_num_filters,
                                                   stack, res_block, self.activation, self.use_bias))
                input_num_filters = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)


def make_resnet20_frn_fn(num_classes, activation=torch.nn.ReLU):
    return make_resnet_fn(
        num_classes, depth=20, normalization_layer=FilterResponseNorm_layer,
        activation=activation)


# def make_medmnist_cnn(n_channels=3, num_classes=10):
#     act = torch.nn.ReLU
#     return nn.Sequential(
#         nn.Conv2d(n_channels, 6, kernel_size=5, padding=2),
#         nn.BatchNorm2d(6),
#         act(),
#         nn.AvgPool2d(2, stride=2, padding=0),
#         nn.Conv2d(6, 16, kernel_size=5, padding=0),
#         nn.BatchNorm2d(16),
#         act(),
#         nn.AvgPool2d(2, stride=2, padding=0),
#         nn.Flatten(),
#         nn.Linear(400, 120),
#         nn.BatchNorm1d(120),
#         act(),
#         nn.Linear(120, 64)
#         # nn.Linear(120, 84)
#         # act(),
#         # nn.Linear(84, num_classes),
#     )