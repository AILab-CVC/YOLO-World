# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from .attn import MaxSigmoidAttnBlock
from .bottleneck import Bottleneck
from .conv import Conv

__all__ = (
    'CSPLayer',
    'MaxSigmoidCSPLayer',
)


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 with_norm: bool = True,
                 with_activation: bool = True) -> None:
        super().__init__()

        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = Conv(
            in_channels, 2 * self.mid_channels, 1, with_norm=with_norm, with_activation=with_activation)
        self.final_conv = Conv(
            (2 + num_blocks) * self.mid_channels, out_channels, 1, with_norm=with_norm, with_activation=with_activation)

        self.blocks = nn.ModuleList(
            Bottleneck(
                self.mid_channels,
                self.mid_channels,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                with_norm=with_norm,
                with_activation=with_activation) for _ in range(num_blocks))

    def forward(self, x: Tensor) -> Tensor:
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))


class MaxSigmoidCSPLayer(CSPLayer):
    """Sigmoid-attention based CSP layer.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            with_norm: bool = True,
            with_activation: bool = True,
            use_einsum: bool = True) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            expand_ratio=expand_ratio,
            num_blocks=num_blocks,
            add_identity=add_identity,
            with_norm=with_norm,
            with_activation=with_activation)

        self.final_conv = Conv(
            (3 + num_blocks) * self.mid_channels, out_channels, 1, with_norm=with_norm, with_activation=with_activation)

        self.attn_block = MaxSigmoidAttnBlock(
            self.mid_channels,
            self.mid_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            with_norm=with_norm,
            use_einsum=use_einsum)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))
