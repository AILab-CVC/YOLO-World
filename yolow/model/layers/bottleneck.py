# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence, Union

from .conv import Conv

__all__ = (
    'Bottleneck',
    'SPPFBottleneck',
)


class Bottleneck(nn.Module):
    """The basic bottleneck block used in Darknet.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 kernel_size: Sequence[int] = (1, 3),
                 padding: Sequence[int] = (0, 1),
                 add_identity: bool = True,
                 with_norm: bool = False,
                 with_activation: bool = True) -> None:
        super().__init__()

        hidden_channels = int(out_channels * expansion)
        assert isinstance(kernel_size, Sequence) and len(kernel_size) == 2

        self.conv1 = Conv(
            in_channels,
            hidden_channels,
            kernel_size[0],
            padding=padding[0],
            with_norm=with_norm,
            with_activation=with_activation)
        self.conv2 = Conv(
            hidden_channels,
            out_channels,
            kernel_size[1],
            stride=1,
            padding=padding[1],
            with_norm=with_norm,
            with_activation=with_activation)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class SPPFBottleneck(nn.Module):
    """Spatial pyramid pooling
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: Union[int, Sequence[int]] = 5,
                 use_conv_first: bool = True,
                 mid_channels_scale: float = 0.5,
                 with_norm: bool = True,
                 with_activation: bool = True):
        super().__init__()

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = Conv(
                in_channels, mid_channels, 1, stride=1, with_norm=with_norm, with_activation=with_activation)
        else:
            mid_channels = in_channels
            self.conv1 = None
        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList(
                [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv2 = Conv(conv2_in_channels, out_channels, 1, with_norm=with_norm, with_activation=with_activation)

    def forward(self, x: Tensor) -> Tensor:
        if self.conv1:
            x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x
