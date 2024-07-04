# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from typing import Tuple

from .layers import Conv, CSPLayer, SPPFBottleneck
from .misc import make_divisible, make_round

__all__ = ('YOLOv8CSPDarknet', )


class YOLOv8CSPDarknet(nn.Module):
    """CSP-Darknet backbone used in YOLOv8.
    """
    arch_settings = {  # in_channels, out_channels, num_blocks, add_identity, use_spp
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 with_norm: bool = True,
                 with_activation: bool = True,
                 norm_eval: bool = False):
        super().__init__()
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        self.arch_settings = self.arch_settings[arch]
        self.num_stages = len(self.arch_settings)

        assert set(out_indices).issubset(i for i in range(len(self.arch_settings) + 1))

        if frozen_stages not in range(-1, len(self.arch_settings) + 1):
            raise ValueError('"frozen_stages" must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval
        self.with_norm = with_norm
        self.with_activation = with_activation

        self.stem = self.build_stem_layer()
        self.layers = ['stem']

        for idx, setting in enumerate(self.arch_settings):
            stage = []
            stage += self.build_stage_layer(idx, setting)
            self.add_module(f'stage{idx + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{idx + 1}')

    def build_stem_layer(self) -> nn.Module:
        return Conv(
            self.input_channels,
            make_divisible(self.arch_settings[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            with_norm=self.with_norm,
            with_activation=self.with_activation)

    def build_stage_layer(self, idx: int, setting: list) -> list:
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = Conv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            with_norm=self.with_norm,
            with_activation=self.with_activation)
        stage.append(conv_layer)
        csp_layer = CSPLayer(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            with_norm=self.with_norm,
            with_activation=self.with_activation)
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                with_norm=self.with_norm,
                with_activation=self.with_activation)
            stage.append(spp)
        return stage

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:  # keep normalization layer frozen
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm):
                    m.eval()

    def forward(self, x: torch.Tensor) -> tuple:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
