# Copyright (c) Tencent Inc. All rights reserved.
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union

__all__ = ('Conv', )


class Conv(nn.Module):
    """A convolution block
    composed of conv/norm/activation layers.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 with_norm: bool = False,
                 with_activation: bool = True,
                 bias: Union[bool, str] = 'auto'):
        super().__init__()
        self.with_norm = with_norm
        self.with_activation = with_activation
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        # build normalization layers
        if self.with_norm:
            self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001)

        # build activation layer
        if self.with_activation:
            self.activate = nn.SiLU(inplace=True)

        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        # fixed order: ('conv', 'norm', 'act')
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_activation:
            x = self.activate(x)
        return x

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

        if self.with_norm:
            nn.init.constant_(self.bn.weight, 1)
            if hasattr(self.conv, 'bias') and self.conv.bias is not None:
                nn.init.constant_(self.bn.bias, 0)
