# Copyright (c) OpenMMLab. All rights reserved.
# Apache License Version 2.0
# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/iou_loss.py

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .utils import weight_reduce_loss, weighted_loss

__all__ = ('CIoULoss', )


@weighted_loss
def ciou_loss(
    pred: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    bbox_format: str = 'xyxy',
    eps: float = 1e-7,
    reduction: str = 'mean',
    avg_factor: Optional[int] = None,
) -> Tensor:

    assert bbox_format in ('xyxy')  # preprocess in dataloader

    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    # Width and height ratio (v)
    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    with torch.no_grad():
        # alpha = (ious > 0.5).float() * v / (1 - ious + v)
        alpha = v / (v - ious + (1 + eps))  # modified

    # CIoU
    ious = ious - (rho2 / c2 + alpha * v)
    ious = ious.clamp(min=-1.0, max=1.0)
    return weight_reduce_loss(1.0 - ious, weight, reduction, avg_factor)


class CIoULoss(nn.Module):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    """

    def __init__(self,
                 bbox_format: str = 'xyxy',
                 eps: float = 1e-7,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:

        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        loss = self.loss_weight * ciou_loss(
            pred,
            target,
            weight,
            bbox_format=self.bbox_format,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss
