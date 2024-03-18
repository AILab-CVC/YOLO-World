# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

_XYWH2XYXY = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                           [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]],
                          dtype=torch.float32)


def select_nms_index(scores: Tensor,
                     boxes: Tensor,
                     nms_index: Tensor,
                     batch_size: int,
                     keep_top_k: int = -1):
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]

    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)
    boxes = boxes[batch_inds, box_inds, ...]
    dets = torch.cat([boxes, scores], dim=1)

    batched_dets = dets.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_template = torch.arange(
        0, batch_size, dtype=batch_inds.dtype, device=batch_inds.device)
    batched_dets = batched_dets.where(
        (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
        batched_dets.new_zeros(1))

    batched_labels = cls_inds.unsqueeze(0).repeat(batch_size, 1)
    batched_labels = batched_labels.where(
        (batch_inds == batch_template.unsqueeze(1)),
        batched_labels.new_ones(1) * -1)

    N = batched_dets.shape[0]

    batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))),
                             1)
    batched_labels = torch.cat((batched_labels, -batched_labels.new_ones(
        (N, 1))), 1)

    _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
    topk_batch_inds = torch.arange(
        batch_size, dtype=topk_inds.dtype,
        device=topk_inds.device).view(-1, 1)
    batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
    batched_labels = batched_labels[topk_batch_inds, topk_inds, ...]
    batched_dets, batched_scores = batched_dets.split([4, 1], 2)
    batched_scores = batched_scores.squeeze(-1)

    num_dets = (batched_scores > 0).sum(1, keepdim=True)
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
    iou_threshold = torch.tensor([iou_threshold])
    score_threshold = torch.tensor([score_threshold])

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
