# Copyright (c) Tencent Inc. All rights reserved.
import json
import os.path as osp
import torch.nn as nn
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

from .clip_text import HuggingCLIPLanguageBackbone
from .data_preprocessor import YOLOWDetDataPreprocessor
from .misc import yolow_dict  # simply replace dict['key'] with dict.key
from .yolo_base import YOLOv8CSPDarknet
from .yolo_world import YOLOWorldDetector
from .yolo_world_backbone import MultiModalYOLOBackbone
from .yolo_world_head import YOLOWorldHead, YOLOWorldHeadModule
from .yolo_world_pafpn import YOLOWorldPAFPN

__all__ = (
    'build_yoloworld_data_preprocessor',
    'build_yolov8_backbone',
    'build_yoloworld_text',
    'build_yoloworld_backbone',
    'build_yoloworld_neck',
    'build_yoloworld_head',
    'build_yoloworld_detector',
)

# default config files for model architectures
# generally we do not need to modify these template files
# you can manually add arguments via `args` when calling functions
CFG_FILES = {
    'n': osp.join(osp.dirname(__file__), 'model_cfgs', 'yoloworld_n.json'),
    's': osp.join(osp.dirname(__file__), 'model_cfgs', 'yoloworld_s.json'),
    'm': osp.join(osp.dirname(__file__), 'model_cfgs', 'yoloworld_m.json'),
    'l': osp.join(osp.dirname(__file__), 'model_cfgs', 'yoloworld_l.json'),
    'x': osp.join(osp.dirname(__file__), 'model_cfgs', 'yoloworld_x.json'),
    'xl': osp.join(osp.dirname(__file__), 'model_cfgs', 'yoloworld_xl.json'),
}


def load_config(size, CfgClass=None, key=None, add_args=None):
    assert size in CFG_FILES.keys(), \
        "YOLO-World only supports the following sizes: [n|s|m|l|x|xl]."
    # read json file into dict
    with open(CFG_FILES[size], 'r') as jf:
        cfg = json.load(jf)
    assert ((key is None) or (key in cfg.keys())), (f'Unknown key: {key}')

    if (CfgClass is not None):
        # read from json WITH default settings
        if (key is None):
            cfg = CfgClass()  # by default
        else:
            cfg = CfgClass(**cfg[key])
    else:
        # read from json WITHOUT default settings
        cfg = yolow_dict(cfg)
        if (key is not None):
            cfg = cfg.key

    # update with manually added config
    if (add_args is not None):
        if (key is not None):
            # add_args = add_args[key]
            add_args = add_args.get(key, dict())
        if isinstance(cfg, CfgClass):
            all_keys = cfg.__dataclass_fields__.keys()
        elif isinstance(cfg, dict):
            all_keys = cfg.keys()
        else:
            raise ValueError(f'Unknown type: {cfg.__class__}')
        # TODO add warning for mismatched kwargs
        filtered_kwargs = {k: add_args[k] for k in all_keys if k in add_args.keys()}
        # print (filtered_kwargs)
        cfg = replace(cfg, **filtered_kwargs)
    return cfg


@dataclass
class YOLOWorldDataPreCfg:
    mean: List = field(default_factory=lambda: [0., 0., 0.])
    std: List = field(default_factory=lambda: [255., 255., 255.])
    pad_size_divisor: int = 1
    pad_value: float = 0
    bgr_to_rgb: bool = True
    rgb_to_bgr: bool = False
    non_blocking: bool = True


def build_yoloworld_data_preprocessor(size: str, args: Optional[dict] = None) -> nn.Module:
    cfg = load_config(size, YOLOWorldDataPreCfg, 'yoloworld_data_preprocessor', args)

    return YOLOWDetDataPreprocessor(
        mean=cfg.mean,
        std=cfg.std,
        pad_size_divisor=cfg.pad_size_divisor,
        pad_value=cfg.pad_value,
        bgr_to_rgb=cfg.bgr_to_rgb,
        rgb_to_bgr=cfg.rgb_to_bgr,
        non_blocking=cfg.non_blocking,
    )


@dataclass
class YOLOv8BackboneCfg:
    arch: str = 'P5'
    last_stage_out_channels: int = 1024  # vary among sizes
    deepen_factor: float = 0.33  # vary among sizes
    widen_factor: float = 0.5  # vary among sizes
    input_channels: int = 3
    out_indices: Tuple[int] = (2, 3, 4)
    frozen_stages: int = -1
    with_norm: bool = True
    with_activation: bool = True
    norm_eval: bool = False


def build_yolov8_backbone(size: str, args: Optional[dict] = None) -> nn.Module:
    """Exp.
    >>> model = build_yolov8_backbone('s')
    """
    cfg = load_config(size, YOLOv8BackboneCfg, 'yolov8_backbone', args)

    return YOLOv8CSPDarknet(
        arch=cfg.arch,
        last_stage_out_channels=cfg.last_stage_out_channels,
        deepen_factor=cfg.deepen_factor,
        widen_factor=cfg.widen_factor,
        input_channels=cfg.input_channels,
        out_indices=cfg.out_indices,
        frozen_stages=cfg.frozen_stages,
        with_norm=cfg.with_norm,  # BN
        with_activation=cfg.with_activation,  # SiLU
        norm_eval=cfg.norm_eval,
    )


@dataclass
class YOLOWorldTextCfg:
    model_name: str = 'openai/clip-vit-base-patch32'
    channels: int = 512  # for `YOLOWorldPAFPN.guide_channels`
    frozen_modules: List = field(default_factory=lambda: ['all'])
    dropout: float = 0.0


def build_yoloworld_text(size: str, args: Optional[dict] = None) -> nn.Module:
    cfg = load_config(size, YOLOWorldTextCfg, 'yoloworld_text', args)

    return HuggingCLIPLanguageBackbone(
        model_name=cfg.model_name,
        frozen_modules=cfg.frozen_modules,
        dropout=cfg.dropout,
    )


@dataclass
class YOLOWorldBackboneCfg:
    frozen_stages: int = -1
    with_text_model: bool = True


def build_yoloworld_backbone(
        size: str,
        image_model: nn.Module,  # from `build_yolov8_backbone`
        text_model: nn.Module,  # from `build_yoloworld_text`
        args: Optional[dict] = None) -> nn.Module:
    cfg = load_config(size, YOLOWorldBackboneCfg, 'yoloworld_backbone', args)

    return MultiModalYOLOBackbone(
        image_model=image_model,
        text_model=text_model,
        frozen_stages=cfg.frozen_stages,
        with_text_model=cfg.with_text_model,
    )


@dataclass
class YOLOWorldNeckCfg:
    in_channels: List = field(default_factory=lambda: [256, 512, 1024])
    out_channels: List = field(default_factory=lambda: [256, 512, 1024])
    embed_channels: List = field(default_factory=lambda: [128, 256, 512])
    num_heads: List = field(default_factory=lambda: [4, 8, 16])
    num_csp_blocks: int = 3
    freeze_all: bool = False
    with_norm: bool = True
    with_activation: bool = True


def build_yoloworld_neck(size: str, args: Optional[dict] = None) -> nn.Module:
    cfg = load_config(size, YOLOWorldNeckCfg, 'yoloworld_neck', args)
    img_cfg = load_config(size, YOLOv8BackboneCfg, 'yolov8_backbone', args)
    text_cfg = load_config(size, YOLOWorldTextCfg, 'yoloworld_text', args)

    return YOLOWorldPAFPN(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        guide_channels=text_cfg.channels,  # determined by text encoder
        embed_channels=cfg.embed_channels,
        num_heads=cfg.num_heads,
        deepen_factor=img_cfg.deepen_factor,
        widen_factor=img_cfg.widen_factor,
        num_csp_blocks=cfg.num_csp_blocks,
        freeze_all=cfg.freeze_all,
        with_norm=cfg.with_norm,
        with_activation=cfg.with_activation,
    )


@dataclass
class YOLOWorldHeadModuleCfg:
    use_bn_head: bool = True
    use_einsum: bool = True
    freeze_all: bool = False
    num_base_priors: int = 1
    featmap_strides: List = field(default_factory=lambda: [8, 16, 32])
    reg_max: int = 16
    with_norm: bool = True
    with_activation: bool = True


def build_yoloworld_head(
        size: str,
        multi_label: bool = True,  # test_cfg
        nms_pre: int = 30000,  # test_cfg
        score_thr: float = 0.001,  # test_cfg
        nms_iou_threshold: float = 0.7,  # test_cfg
        max_per_img: int = 300,  # test_cfg
        args: Optional[dict] = None) -> nn.Module:

    cfg = load_config(size, YOLOWorldHeadModuleCfg, 'yoloworld_head_module', args)
    text_cfg = load_config(size, YOLOWorldTextCfg, 'yoloworld_text', args)
    yolo_cfg = load_config(size, YOLOv8BackboneCfg, 'yolov8_backbone', args)
    neck_cfg = load_config(size, YOLOWorldNeckCfg, 'yoloworld_neck', args)
    det_cfg = load_config(size, YOLOWorldDetectorCfg, 'yoloworld_detector', args)

    # `test_cfg` is not kind of model arch
    # so we did not define it in the json file
    # determine the arguments when calling `build_yoloworld_head()`
    test_cfg = yolow_dict(
        # The config of multi-label for multi-class prediction.
        multi_label=multi_label,
        # The number of boxes before NMS
        nms_pre=nms_pre,
        score_thr=score_thr,  # Threshold to filter out boxes.
        nms=yolow_dict(type='nms', iou_threshold=nms_iou_threshold),  # NMS type and threshold
        max_per_img=max_per_img)  # Max number of detections of each image

    return YOLOWorldHead(
        YOLOWorldHeadModule(
            num_classes=det_cfg.num_train_classes,
            in_channels=neck_cfg.in_channels,  # determined by neck
            embed_dims=text_cfg.channels,  # determined by text encoder
            use_bn_head=cfg.use_bn_head,
            use_einsum=cfg.use_einsum,
            freeze_all=cfg.freeze_all,
            widen_factor=yolo_cfg.widen_factor,  # determined by yolov8
            num_base_priors=cfg.num_base_priors,
            featmap_strides=cfg.featmap_strides,
            reg_max=cfg.reg_max,
            with_norm=cfg.with_norm,
            with_activation=cfg.with_activation,
        ),
        test_cfg=test_cfg,
    )


@dataclass
class YOLOWorldDetectorCfg:
    mm_neck: bool = True
    use_syncbn: bool = True
    num_train_classes: int = 80
    num_test_classes: int = 80


def build_yoloworld_detector(size: str,
                             backbone: nn.Module,
                             neck: nn.Module,
                             bbox_head: nn.Module,
                             data_preprocessor: Optional[nn.Module] = None,
                             args: Optional[dict] = None) -> nn.Module:
    cfg = load_config(size, YOLOWorldDetectorCfg, 'yoloworld_detector', args)

    return YOLOWorldDetector(
        backbone=backbone,
        neck=neck,
        bbox_head=bbox_head,
        mm_neck=cfg.mm_neck,
        num_train_classes=cfg.num_train_classes,
        num_test_classes=cfg.num_test_classes,
        data_preprocessor=data_preprocessor,
        use_syncbn=cfg.use_syncbn,
    )
