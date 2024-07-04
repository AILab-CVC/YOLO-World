# Copyright (c) OpenMMLab. All rights reserved.
# Apache License Version 2.0
# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/coders/base_bbox_coder.py

import torch
from torch import Tensor
from typing import Optional, Sequence, Union

__all__ = ('DistancePointBBoxCoder', )


def bbox2distance(points: Tensor, bbox: Tensor, max_dis: Optional[float] = None, eps: float = 0.1) -> Tensor:
    """Decode bounding box based on distances.
    """
    left = points[..., 0] - bbox[..., 0]
    top = points[..., 1] - bbox[..., 1]
    right = bbox[..., 2] - points[..., 0]
    bottom = bbox[..., 3] - points[..., 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def distance2bbox(points: Tensor,
                  distance: Tensor,
                  max_shape: Optional[Union[Sequence[int], Tensor, Sequence[Sequence[int]]]] = None) -> Tensor:
    """Decode distance prediction to bounding box.
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            raise NotImplementedError
        #     # TODO: delete
        #     from mmdet.core.export import dynamic_clip_for_onnx
        #     x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
        #     bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        #     return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape], dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes


class DistancePointBBoxCoder:
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.
    """

    # The size of the last of dimension of the encoded tensor.
    encode_size = 4

    def __init__(self, clip_border: Optional[bool] = True, use_box_type: bool = False) -> None:
        self.use_box_type = use_box_type
        self.clip_border = clip_border

    def encode(
            self,
            points: Tensor,
            gt_bboxes: Tensor,  # modified
            max_dis: float = 16.,
            eps: float = 0.01) -> Tensor:
        """Encode bounding box to distances.
        """
        assert points.size(-2) == gt_bboxes.size(-2)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        return bbox2distance(points, gt_bboxes, max_dis, eps)

    def decode(
            self,
            points: Tensor,
            pred_bboxes: Tensor,
            stride: torch.Tensor,  # modified
            max_shape: Optional[Union[Sequence[int], Tensor, Sequence[Sequence[int]]]] = None) -> Tensor:
        """Decode distance prediction to bounding box.
        """
        assert points.size(-2) == pred_bboxes.size(-2)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        if self.clip_border is False:
            max_shape = None

        bboxes = distance2bbox(points, pred_bboxes * stride[None, :, None], max_shape)
        return bboxes
