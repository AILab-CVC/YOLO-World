# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

_XYWH2XYXY = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                           [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]],
                          dtype=torch.float32)


class TRTEfficientNMSop(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        background_class: int = -1,
        box_coding: int = 0,
        iou_threshold: float = 0.45,
        max_output_boxes: int = 100,
        plugin_version: str = '1',
        score_activation: int = 0,
        score_threshold: float = 0.25,
    ):
        batch_size, _, num_classes = scores.shape
        num_det = torch.randint(
            0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes: Tensor,
                 scores: Tensor,
                 background_class: int = -1,
                 box_coding: int = 0,
                 iou_threshold: float = 0.45,
                 max_output_boxes: int = 100,
                 plugin_version: str = '1',
                 score_activation: int = 0,
                 score_threshold: float = 0.25):
        out = g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4)
        num_det, det_boxes, det_scores, det_classes = out
        return num_det, det_boxes, det_scores, det_classes


class TRTbatchedNMSop(torch.autograd.Function):
    """TensorRT NMS operation."""

    @staticmethod
    def forward(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        plugin_version: str = '1',
        shareLocation: int = 1,
        backgroundLabelId: int = -1,
        numClasses: int = 80,
        topK: int = 1000,
        keepTopK: int = 100,
        scoreThreshold: float = 0.25,
        iouThreshold: float = 0.45,
        isNormalized: int = 0,
        clipBoxes: int = 0,
        scoreBits: int = 16,
        caffeSemantics: int = 1,
    ):
        batch_size, _, numClasses = scores.shape
        num_det = torch.randint(
            0, keepTopK, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, keepTopK, 4)
        det_scores = torch.randn(batch_size, keepTopK)
        det_classes = torch.randint(0, numClasses,
                                    (batch_size, keepTopK)).float()
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes: Tensor,
        scores: Tensor,
        plugin_version: str = '1',
        shareLocation: int = 1,
        backgroundLabelId: int = -1,
        numClasses: int = 80,
        topK: int = 1000,
        keepTopK: int = 100,
        scoreThreshold: float = 0.25,
        iouThreshold: float = 0.45,
        isNormalized: int = 0,
        clipBoxes: int = 0,
        scoreBits: int = 16,
        caffeSemantics: int = 1,
    ):
        out = g.op(
            'TRT::BatchedNMSDynamic_TRT',
            boxes,
            scores,
            shareLocation_i=shareLocation,
            plugin_version_s=plugin_version,
            backgroundLabelId_i=backgroundLabelId,
            numClasses_i=numClasses,
            topK_i=topK,
            keepTopK_i=keepTopK,
            scoreThreshold_f=scoreThreshold,
            iouThreshold_f=iouThreshold,
            isNormalized_i=isNormalized,
            clipBoxes_i=clipBoxes,
            scoreBits_i=scoreBits,
            caffeSemantics_i=caffeSemantics,
            outputs=4)
        num_det, det_boxes, det_scores, det_classes = out
        return num_det, det_boxes, det_scores, det_classes


def _efficient_nms(
    boxes: Tensor,
    scores: Tensor,
    max_output_boxes_per_class: int = 1000,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = 100,
    box_coding: int = 0,
):
    """Wrapper for `efficient_nms` with TensorRT.
    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        box_coding (int): Bounding boxes format for nms.
            Defaults to 0 means [x1, y1 ,x2, y2].
            Set to 1 means [x, y, w, h].
    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
        (num_det, det_boxes, det_scores, det_classes),
        `num_det` of shape [N, 1]
        `det_boxes` of shape [N, num_det, 4]
        `det_scores` of shape [N, num_det]
        `det_classes` of shape [N, num_det]
    """
    num_det, det_boxes, det_scores, det_classes = TRTEfficientNMSop.apply(
        boxes, scores, -1, box_coding, iou_threshold, keep_top_k, '1', 0,
        score_threshold)
    return num_det, det_boxes, det_scores, det_classes


def _batched_nms(
    boxes: Tensor,
    scores: Tensor,
    max_output_boxes_per_class: int = 1000,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = 100,
    box_coding: int = 0,
):
    """Wrapper for `efficient_nms` with TensorRT.
    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        box_coding (int): Bounding boxes format for nms.
            Defaults to 0 means [x1, y1 ,x2, y2].
            Set to 1 means [x, y, w, h].
    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
        (num_det, det_boxes, det_scores, det_classes),
        `num_det` of shape [N, 1]
        `det_boxes` of shape [N, num_det, 4]
        `det_scores` of shape [N, num_det]
        `det_classes` of shape [N, num_det]
    """
    if box_coding == 1:
        boxes = boxes @ (_XYWH2XYXY.to(boxes.device))
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    _, _, numClasses = scores.shape

    num_det, det_boxes, det_scores, det_classes = TRTbatchedNMSop.apply(
        boxes, scores, '1', 1, -1, int(numClasses), min(pre_top_k, 4096),
        keep_top_k, score_threshold, iou_threshold, 0, 0, 16, 1)

    det_classes = det_classes.int()
    return num_det, det_boxes, det_scores, det_classes


def efficient_nms(*args, **kwargs):
    """Wrapper function for `_efficient_nms`."""
    return _efficient_nms(*args, **kwargs)


def batched_nms(*args, **kwargs):
    """Wrapper function for `_batched_nms`."""
    return _batched_nms(*args, **kwargs)
