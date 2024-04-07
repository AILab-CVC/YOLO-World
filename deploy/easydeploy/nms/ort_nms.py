# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

_XYWH2XYXY = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                           [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]],
                          dtype=torch.float32)


def sort_nms_index(nms_index, scores, keep_top_k=-1):
    """
    first sort the nms_index by batch, and then sort by score in every image result, final apply keep_top_k strategy. In the process, we can also get the number of detections for each image: num_dets
    """
    # first sort by batch index to make sure that the same batch index is together
    device = nms_index.device
    nms_index_indices = torch.argsort(nms_index[:, 0], dim=0).to(device)
    nms_index = nms_index[nms_index_indices]

    scores = scores[nms_index[:, 0], nms_index[:, 1], nms_index[:, 2]]
    batch_inds = nms_index[:, 0]

    # Get the number of detections for each image
    _, num_dets = torch.unique(batch_inds, return_counts=True)
    num_dets = num_dets.to(device)
    # Calculate the sum from front to back
    cumulative_sum = torch.cumsum(num_dets, dim=0).to(device)
    # add initial value 0
    cumulative_sum = torch.cat((torch.tensor([0]).to(device), cumulative_sum))
    for i in range(len(num_dets)):
        start = cumulative_sum[i]
        end = cumulative_sum[i + 1]
        # sort by score in every batch
        block_idx = torch.argsort(scores[start:end], descending=True).to(device)
        nms_index[start:end] = nms_index[start:end][block_idx]
        if keep_top_k > 0 and end - start > keep_top_k:
            # delete lines from start+keep_top_k to end to keep only top k
            nms_index = torch.cat(
                (nms_index[: start + keep_top_k], nms_index[end:]), dim=0
            )
            num_dets[i] -= end - start - keep_top_k
            cumulative_sum -= end - start - keep_top_k
    return nms_index, num_dets


def select_nms_index(
    scores: Tensor,
    boxes: Tensor,
    nms_index: Tensor,
    batch_size: int,
    keep_top_k: int = -1,
):
    if nms_index.numel() == 0:
        return torch.empty(0), torch.empty(0, 4), torch.empty(0), torch.empty(0)
    nms_index, num_dets = sort_nms_index(nms_index, scores, keep_top_k)
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]

    # according to the nms_index to get the scores,boxes and labels
    batched_scores = scores[batch_inds, cls_inds, box_inds]
    batched_dets = boxes[batch_inds, box_inds, ...]
    batched_labels = cls_inds

    return num_dets, batched_dets, batched_scores, batched_labels


class ONNXNMSop(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        max_output_boxes_per_class: Tensor = torch.tensor([100]),
        iou_threshold: Tensor = torch.tensor([0.5]),
        score_threshold: Tensor = torch.tensor([0.05])
    ) -> Tensor:
        device = boxes.device
        batch = scores.shape[0]
        num_det = 20
        batches = torch.randint(0, batch, (num_det, )).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det, ), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]],
                                     0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)

        return selected_indices

    @staticmethod
    def symbolic(
            g,
            boxes: Tensor,
            scores: Tensor,
            max_output_boxes_per_class: Tensor = torch.tensor([100]),
            iou_threshold: Tensor = torch.tensor([0.5]),
            score_threshold: Tensor = torch.tensor([0.05]),
    ):
        return g.op(
            'NonMaxSuppression',
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            outputs=1)


def onnx_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    max_output_boxes_per_class: int = 100,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = 100,
    box_coding: int = 0,
):
    max_output_boxes_per_class = torch.tensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold]).to(boxes.device)
    score_threshold = torch.tensor([score_threshold]).to(boxes.device)

    batch_size, _, _ = scores.shape
    if box_coding == 1:
        boxes = boxes @ (_XYWH2XYXY.to(boxes.device))
    scores = scores.transpose(1, 2).contiguous()
    selected_indices = ONNXNMSop.apply(boxes, scores,
                                       max_output_boxes_per_class,
                                       iou_threshold, score_threshold)

    num_dets, batched_dets, batched_scores, batched_labels = select_nms_index(
        scores, boxes, selected_indices, batch_size, keep_top_k=keep_top_k)

    return num_dets, batched_dets, batched_scores, batched_labels.to(
        torch.int32)
