# Copyright (c) Tencent Inc. All rights reserved.
import torch
from torch import Tensor
from mmyolo.registry import TASK_UTILS
from mmyolo.models.task_modules.assigners import BatchTaskAlignedAssigner
from mmyolo.models.task_modules.assigners.utils import select_highest_overlaps

@TASK_UTILS.register_module()
class YOLOWorldSegAssigner(BatchTaskAlignedAssigner):

    def __init__(self,
                 num_classes: int,
                 topk: int = 13,
                 alpha: float = 1,
                 beta: float = 6,
                 eps: float = 1e-7,
                 use_ciou: bool = False):
        super().__init__(num_classes, topk, alpha, beta, eps, use_ciou)

    @torch.no_grad()
    def forward(
        self,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        priors: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pad_bbox_flag: Tensor,
    ) -> dict:
        """Assign gt to bboxes.

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)
        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bboxes,
                shape(batch_size, num_priors, num_classes)
            priors (Tensor): Model priors,  shape (num_priors, 4)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
        Returns:
            assigned_result (dict) Assigned result:
                assigned_labels (Tensor): Assigned labels,
                    shape(batch_size, num_priors)
                assigned_bboxes (Tensor): Assigned boxes,
                    shape(batch_size, num_priors, 4)
                assigned_scores (Tensor): Assigned scores,
                    shape(batch_size, num_priors, num_classes)
                fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                    shape(batch_size, num_priors)
        """
        # (num_priors, 4) -> (num_priors, 2)
        priors = priors[:, :2]

        batch_size = pred_scores.size(0)
        num_gt = gt_bboxes.size(1)

        assigned_result = {
            'assigned_labels':
            gt_bboxes.new_full(pred_scores[..., 0].shape, self.num_classes),
            'assigned_bboxes':
            gt_bboxes.new_full(pred_bboxes.shape, 0),
            'assigned_scores':
            gt_bboxes.new_full(pred_scores.shape, 0),
            'fg_mask_pre_prior':
            gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
        }

        if num_gt == 0:
            return assigned_result

        pos_mask, alignment_metrics, overlaps = self.get_pos_mask(
            pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes,
            pad_bbox_flag, batch_size, num_gt)

        (assigned_gt_idxs, fg_mask_pre_prior,
         pos_mask) = select_highest_overlaps(pos_mask, overlaps, num_gt)

        # assigned target
        assigned_labels, assigned_bboxes, assigned_scores = self.get_targets(
            gt_labels, gt_bboxes, assigned_gt_idxs, fg_mask_pre_prior,
            batch_size, num_gt)

        # normalize
        alignment_metrics *= pos_mask
        pos_align_metrics = alignment_metrics.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * pos_mask).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (
            alignment_metrics * pos_overlaps /
            (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
        assigned_scores = assigned_scores * norm_align_metric

        assigned_result['assigned_labels'] = assigned_labels
        assigned_result['assigned_bboxes'] = assigned_bboxes
        assigned_result['assigned_scores'] = assigned_scores
        assigned_result['fg_mask_pre_prior'] = fg_mask_pre_prior.bool()
        assigned_result['assigned_gt_idxs'] = assigned_gt_idxs
        return assigned_result
