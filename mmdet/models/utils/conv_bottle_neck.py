import torch
import torch.nn as nn
from torch.nn import init


def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=stride,
                     padding=(0, 1), groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride,
                     padding=(1, 1), groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet_text_50(nn.Module):
    def __init__(self, in_dim, out_dim, zero_init_residual=False, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_text_50, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = in_dim

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = conv1x1(self.inplanes, out_dim//2)
        self.bn1 = norm_layer(out_dim//2)
        self.relu = nn.ReLU(inplace=True)

        downsample = nn.Sequential(
            conv1x1(out_dim//2, out_dim),
            norm_layer(out_dim),
        )

        # 3, 4, 6, 3
        self.branch1 = nn.Sequential(
            Bottleneck(inplanes=out_dim//2, planes=out_dim, width=out_dim//4, downsample=downsample),
            Bottleneck(inplanes=out_dim, planes=out_dim, width=out_dim//4),
            Bottleneck(inplanes=out_dim, planes=out_dim, width=out_dim//4)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(2).contiguous()
        x1 = self.conv1(x)  # 1024 1 64
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x21 = self.branch1(x1)
        return x1, x21


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 width=64, dilation=1, norm_layer=None, conv3=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if conv3:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv1x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 width=64, dilation=1, norm_layer=None, conv3=False):
        super(Bottleneck1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        if conv3:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        else:
            self.conv2 = nn.Conv1d(width, width, 1, stride=stride, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv1d(width, planes, 1)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


class conv(nn.Module):
    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.LeakyReLU(0.25, inplace=True)]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x