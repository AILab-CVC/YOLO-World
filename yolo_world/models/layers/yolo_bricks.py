# Copyright (c) Tencent Inc. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Linear
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmyolo.models.layers import CSPLayerWithTwoConv


@MODELS.register_module()
class MaxSigmoidAttnBlock(BaseModule):
    """Max Sigmoid attention block."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.guide_fc = Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        B, _, H, W = x.shape

        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale

        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)
        return x


@MODELS.register_module()
class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""
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
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = MaxSigmoidAttnBlock(self.mid_channels,
                                              self.mid_channels,
                                              guide_channels=guide_channels,
                                              embed_channels=embed_channels,
                                              num_heads=num_heads,
                                              with_scale=with_scale,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))


@MODELS.register_module()
class ImagePoolingAttentionModule(nn.Module):
    def __init__(self,
                 image_channels: List[int],
                 text_channels: int,
                 embed_channels: int,
                 with_scale: bool = False,
                 num_feats: int = 3,
                 num_heads: int = 8,
                 pool_size: int = 3):
        super().__init__()

        self.text_channels = text_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.num_feats = num_feats
        self.head_channels = embed_channels // num_heads
        self.pool_size = pool_size

        if with_scale:
            self.scale = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        else:
            self.scale = 1.0
        self.projections = nn.ModuleList([
            ConvModule(in_channels, embed_channels, 1, act_cfg=None)
            for in_channels in image_channels
        ])
        self.query = nn.Sequential(nn.LayerNorm(text_channels),
                                   Linear(text_channels, embed_channels))
        self.key = nn.Sequential(nn.LayerNorm(embed_channels),
                                 Linear(embed_channels, embed_channels))
        self.value = nn.Sequential(nn.LayerNorm(embed_channels),
                                   Linear(embed_channels, embed_channels))
        self.proj = Linear(embed_channels, text_channels)

        self.image_pools = nn.ModuleList([
            nn.AdaptiveMaxPool2d((pool_size, pool_size))
            for _ in range(num_feats)
        ])

    def forward(self, text_features, image_features):
        B = image_features[0].shape[0]
        assert len(image_features) == self.num_feats
        num_patches = self.pool_size**2
        mlvl_image_features = [
            pool(proj(x)).view(B, -1, num_patches)
            for (x, proj, pool
                 ) in zip(image_features, self.projections, self.image_pools)
        ]
        mlvl_image_features = torch.cat(mlvl_image_features,
                                        dim=-1).transpose(1, 2)
        q = self.query(text_features)
        k = self.key(mlvl_image_features)
        v = self.value(mlvl_image_features)

        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)

        attn_weight = torch.einsum('bnmc,bkmc->bmnk', q, k)
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)

        x = torch.einsum('bmnk,bkmc->bnmc', attn_weight, v)
        x = self.proj(x.reshape(B, -1, self.embed_channels))
        return x * self.scale + text_features


@MODELS.register_module()
class VanillaSigmoidBlock(BaseModule):
    """Sigmoid attention block."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads

        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x = self.project_conv(x)
        x = x * x.sigmoid()
        return x


@MODELS.register_module()
class EfficientCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""
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
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = VanillaSigmoidBlock(self.mid_channels,
                                              self.mid_channels,
                                              guide_channels=guide_channels,
                                              embed_channels=embed_channels,
                                              num_heads=num_heads,
                                              with_scale=with_scale,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))
