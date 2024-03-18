# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torch import Tensor


def yolov5_bbox_decoder(priors: Tensor, bbox_preds: Tensor,
                        stride: Tensor) -> Tensor:
    bbox_preds = bbox_preds.sigmoid()

    x_center = (priors[..., 0] + priors[..., 2]) * 0.5
    y_center = (priors[..., 1] + priors[..., 3]) * 0.5
    w = priors[..., 2] - priors[..., 0]
    h = priors[..., 3] - priors[..., 1]

    x_center_pred = (bbox_preds[..., 0] - 0.5) * 2 * stride + x_center
    y_center_pred = (bbox_preds[..., 1] - 0.5) * 2 * stride + y_center
    w_pred = (bbox_preds[..., 2] * 2)**2 * w
    h_pred = (bbox_preds[..., 3] * 2)**2 * h

    decoded_bboxes = torch.stack(
        [x_center_pred, y_center_pred, w_pred, h_pred], dim=-1)

    return decoded_bboxes


def rtmdet_bbox_decoder(priors: Tensor, bbox_preds: Tensor,
                        stride: Optional[Tensor]) -> Tensor:
    stride = stride[None, :, None]
    bbox_preds *= stride
    tl_x = (priors[..., 0] - bbox_preds[..., 0])
    tl_y = (priors[..., 1] - bbox_preds[..., 1])
    br_x = (priors[..., 0] + bbox_preds[..., 2])
    br_y = (priors[..., 1] + bbox_preds[..., 3])
    decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    return decoded_bboxes


def yolox_bbox_decoder(priors: Tensor, bbox_preds: Tensor,
                       stride: Optional[Tensor]) -> Tensor:
    stride = stride[None, :, None]
    xys = (bbox_preds[..., :2] * stride) + priors
    whs = bbox_preds[..., 2:].exp() * stride
    decoded_bboxes = torch.cat([xys, whs], -1)
    return decoded_bboxes
