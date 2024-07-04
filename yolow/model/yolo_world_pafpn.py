# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Union

from .layers import Conv, MaxSigmoidCSPLayer
from .misc import make_divisible, make_round

__all__ = ('YOLOWorldPAFPN', )


class YOLOWorldPAFPN(nn.Module):
    """Path Aggregation Network used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 with_norm: bool = True,
                 with_activation: bool = True) -> None:
        super().__init__()
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.num_csp_blocks = num_csp_blocks

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.upsample_feats_cat_first = True
        self.freeze_all = freeze_all
        self.with_norm = with_norm
        self.with_activation = with_activation

        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(self.build_reduce_layer(idx))

        # build top-down blocks
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.upsample_layers.append(self.build_upsample_layer(idx))
            self.top_down_layers.append(self.build_top_down_layer(idx))

        # build bottom-up blocks
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_layers.append(self.build_downsample_layer(idx))
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        self.out_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.out_layers.append(self.build_out_layer(idx))

    def _freeze_all(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def build_reduce_layer(self, idx: int) -> nn.Module:
        return nn.Identity()

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_downsample_layer(self, idx: int) -> nn.Module:
        return Conv(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            with_norm=self.with_norm,
            with_activation=self.with_activation)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        return MaxSigmoidCSPLayer(
            in_channels=make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]), self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx - 1], self.widen_factor),
            guide_channels=self.guide_channels,
            embed_channels=make_round(self.embed_channels[idx - 1], self.widen_factor),
            num_heads=make_round(self.num_heads[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            with_norm=self.with_norm,
            with_activation=self.with_activation)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        return MaxSigmoidCSPLayer(
            in_channels=make_divisible((self.out_channels[idx] + self.out_channels[idx + 1]), self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx + 1], self.widen_factor),
            guide_channels=self.guide_channels,
            embed_channels=make_round(self.embed_channels[idx + 1], self.widen_factor),
            num_heads=make_round(self.num_heads[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            with_norm=self.with_norm,
            with_activation=self.with_activation)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat([downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)
