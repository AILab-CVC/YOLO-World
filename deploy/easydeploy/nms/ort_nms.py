# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
from torchvision.ops import batched_nms

_XYWH2XYXY = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                           [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]],
                          dtype=torch.float32)


def sort_nms_index(nms_index, scores, batch_size, keep_top_k=-1):
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
    num_dets = torch.bincount(batch_inds,minlength=batch_size).to(device)
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
    nms_index, num_dets = sort_nms_index(nms_index, scores, batch_size, keep_top_k)
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]

    # according to the nms_index to get the scores,boxes and labels
    batched_scores = scores[batch_inds, cls_inds, box_inds]
    batched_dets = boxes[batch_inds, box_inds, ...]
    batched_labels = cls_inds

    return num_dets, batched_dets, batched_scores, batched_labels


def construct_indice(batch_idx, select_bbox_idxs, class_idxs, original_idxs):
    num_bbox = len(select_bbox_idxs)
    class_idxs = class_idxs[select_bbox_idxs]
    indice = torch.zeros((num_bbox, 3), dtype=torch.int32).to(select_bbox_idxs.device)
    # batch_idx
    indice[:, 0] = batch_idx
    # class_idxs
    indice[:, 1] = class_idxs
    # select_bbox_idxs
    indice[:, 2] = original_idxs[select_bbox_idxs]
    return indice


def filter_max_boxes_per_class(
    select_bbox_idxs, class_idxs, max_output_boxes_per_class
):
    class_counts = {}  #  used to track the count of each class

    filtered_select_bbox_idxs = []
    filtered_max_class_idxs = []

    for bbox_idx, class_idx in zip(select_bbox_idxs, class_idxs):
        class_count = class_counts.get(
            class_idx.item(), 0
        )  #  Get the count of the current class, or return 0 if it does not exist
        if class_count < max_output_boxes_per_class:
            filtered_select_bbox_idxs.append(bbox_idx)
            filtered_max_class_idxs.append(class_idx)
            class_counts[class_idx.item()] = class_count + 1
    return torch.tensor(filtered_select_bbox_idxs), torch.tensor(
        filtered_max_class_idxs
    )


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
        """
        Non-Maximum Suppression (NMS) implementation.

        Args:
            boxes (Tensor): Bounding boxes of shape (batch_size, num_boxes, 4).
            scores (Tensor): Confidence scores of shape (batch_size, num_classes, num_boxes).
            max_output_boxes_per_class (Tensor): Maximum number of output boxes per class.
            iou_threshold (Tensor): IoU threshold for NMS.
            score_threshold (Tensor): Confidence score threshold.

        Returns:
            Tensor: Selected indices of shape (num_det, 3).first value is batch index, second value is class index, third value is box index
        """
        device = boxes.device
        batch_size, num_classes, num_boxes = scores.shape
        selected_indices = []
        for batch_idx in range(batch_size):
            boxes_per_image = boxes[batch_idx]
            scores_per_image = scores[batch_idx]

            # If no boxes in this image, continue to the next image
            if boxes_per_image.numel() == 0:
                continue

            # for one box, only exist one class,so use torch.max to get the max score and class index
            scores_per_image, class_idxs = torch.max(scores_per_image, dim=0)
            # Apply score threshold before batched_nms bacause nms operation is time expensive
            keep_idxs = scores_per_image > score_threshold
            if not torch.any(keep_idxs):
                # If no boxes left after applying score threshold, continue to the next image
                continue

            boxes_per_image = boxes_per_image[keep_idxs]
            scores_per_image = scores_per_image[keep_idxs]
            class_idxs = class_idxs[keep_idxs]

            #  The purpose of original_idxs is we want to return the indexs to the original input data instead of the filtered.
            original_idxs = torch.arange(num_boxes, device=device)[keep_idxs]
            # reference: https://pytorch.org/vision/main/generated/torchvision.ops.batched_nms.html
            select_bbox_idxs = batched_nms(
                boxes_per_image, scores_per_image, class_idxs, iou_threshold
            )
            if (
                select_bbox_idxs.shape[0] > max_output_boxes_per_class
            ):  # If the boxes detected by all classes together are less than max_output_boxes_per_class, then there is no need to filter
                select_bbox_idxs, _ = filter_max_boxes_per_class(
                    select_bbox_idxs,
                    class_idxs[select_bbox_idxs],
                    max_output_boxes_per_class,
                )
            selected_indice = construct_indice(
                batch_idx, select_bbox_idxs, class_idxs, original_idxs
            )
            selected_indices.append(selected_indice)
        if len(selected_indices) == 0:
            return torch.tensor([], device=device)
        selected_indices = torch.cat(selected_indices, dim=0)
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
