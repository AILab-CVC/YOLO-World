# Copyright (c) OpenMMLab. All rights reserved.
# Apache-2.0 license
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc
from functools import partial
from torch import Tensor
from torchvision.ops import boxes as box_ops
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from ..dist_utils import get_dist_info, get_rank, get_world_size

__all__ = (  # TODO remove useless ones
    'get_world_size',
    'get_rank',
    'get_dist_info',
    'yolow_dict',
    'gt_instances_preprocess',
    'is_seq_of',
    'is_list_of',
    'stack_batch',
    'make_divisible',
    'make_round',
    'multi_apply',
    'unpack_gt_instances',
    'filter_scores_and_topk',
    'get_prior_xy_info',
    'scale_boxes',
    'get_box_wh',
    'nms',
    'batched_nms',
    'revert_sync_batchnorm',
)


class yolow_dict(dict):

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def gt_instances_preprocess(input_data: Union[Tensor, Sequence], total_batches: int) -> Tensor:
    """Process ground truth data based on batch size.
    """
    if isinstance(input_data, Sequence):
        max_bbox_length = max([len(single_instance) for single_instance in input_data])
        instance_collection = []
        for index, single_instance in enumerate(input_data):
            bbox_data = single_instance.bboxes
            label_data = single_instance.labels
            assert isinstance(bbox_data, Tensor)
            bbox_dimension = bbox_data.size(-1)
            instance_collection.append(torch.cat((label_data[:, None], bbox_data), dim=-1))

            if bbox_data.shape[0] < max_bbox_length:
                padding_data = bbox_data.new_full([max_bbox_length - bbox_data.shape[0], bbox_dimension + 1], 0)
                instance_collection[index] = torch.cat((instance_collection[index], padding_data), dim=0)

        return torch.stack(instance_collection)
    else:
        bbox_dimension = input_data.size(-1) - 2
        if len(input_data) > 0:
            image_idx = input_data[:, 0]
            max_bbox_length = image_idx.unique(return_counts=True)[1].max()
            batched_data = torch.zeros((total_batches, max_bbox_length, bbox_dimension + 1),
                                         dtype=input_data.dtype,
                                         device=input_data.device)

            for i in range(total_batches):
                matching_idx = image_idx == i
                gt_count = matching_idx.sum()
                if gt_count:
                    batched_data[i, :gt_count] = input_data[matching_idx, 1:]
        else:
            batched_data = torch.zeros((total_batches, 0, bbox_dimension + 1),
                                         dtype=input_data.dtype,
                                         device=input_data.device)

        return batched_data


def is_seq_of(seq: Any, expected_type: Union[Type, tuple], seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.
    """
    assert isinstance(tensor_list, list), (f'Expected input type to be list, but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({tensor.ndim
                for tensor in tensor_list}) == 1, (f'Expected the dimensions of all tensors must be the same, '
                                                   f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor([tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)


def make_divisible(x: float, widen_factor: float = 1.0, divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unpack_gt_instances(batch_data_samples: list) -> tuple:
    """Unpack ``gt_instances``, ``gt_instances_ignore`` and ``img_metas`` based
    on ``batch_data_samples``
    """
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if 'ignored_instances' in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)
    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results


def get_prior_xy_info(index: int, num_base_priors: int, featmap_sizes: int) -> Tuple[int, int, int]:
    """Get prior index and xy index in feature map by flatten index."""
    _, featmap_w = featmap_sizes
    priors = index % num_base_priors
    xy_index = index // num_base_priors
    grid_y = xy_index // featmap_w
    grid_x = xy_index % featmap_w
    return priors, grid_x, grid_y


def scale_boxes(boxes: Union[Tensor, dict], scale_factor: Tuple[float, float]) -> Union[Tensor, dict]:
    """Scale boxes with type of tensor or box type.
    """
    if isinstance(boxes, dict):
        boxes.rescale_(scale_factor)
        return boxes
    else:
        # Tensor boxes will be treated as horizontal boxes
        repeat_num = int(boxes.size(-1) / 2)
        scale_factor = boxes.new_tensor(scale_factor).repeat((1, repeat_num))
        return boxes * scale_factor


def get_box_wh(boxes: Union[Tensor, dict]) -> Tuple[Tensor, Tensor]:
    """Get the width and height of boxes with type of tensor or box type.
    """
    if isinstance(boxes, dict):
        w = boxes.widths
        h = boxes.heights
    else:
        # Tensor boxes will be treated as horizontal boxes by defaults
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
    return w, h


# This function is modified from: https://github.com/pytorch/vision/
class NMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, bboxes: Tensor, scores: Tensor, iou_threshold: float, offset: int, score_threshold: float,
                max_num: int) -> Tensor:
        is_filtering_by_score = score_threshold > 0
        if is_filtering_by_score:
            valid_mask = scores > score_threshold
            bboxes, scores = bboxes[valid_mask], scores[valid_mask]
            valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)

        # inds = ext_module.nms(
        #     bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)
        inds = box_ops.batched_nms(bboxes.float(), scores, torch.ones(bboxes.size(0)), iou_threshold)

        if max_num > 0:
            inds = inds[:max_num]
        if is_filtering_by_score:
            inds = valid_inds[inds]
        return inds


def nms(boxes: Union[Tensor, np.ndarray],
        scores: Union[Tensor, np.ndarray],
        iou_threshold: float,
        offset: int = 0,
        score_threshold: float = 0,
        max_num: int = -1) -> Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]]:
    assert isinstance(boxes, (Tensor, np.ndarray))
    assert isinstance(scores, (Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    inds = NMSop.apply(boxes, scores, iou_threshold, offset, score_threshold, max_num)

    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds


def batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor,
                nms_cfg: Optional[Dict],
                class_agnostic: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]], dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_op = nms_cfg_.pop('type', 'nms')
    if isinstance(nms_op, str):
        nms_op = eval(nms_op)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


class _BatchNormXd(nn.modules.batchnorm._BatchNorm):
    """A general BatchNorm layer without input dimension check.

    Reproduced from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)
    The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
    is `_check_input_dim` that is designed for tensor sanity checks.
    The check has been bypassed in this class for the convenience of converting
    SyncBatchNorm.
    """

    def _check_input_dim(self, input: torch.Tensor):
        return


def revert_sync_batchnorm(module: nn.Module) -> nn.Module:
    """Helper function to convert all `SyncBatchNorm` (SyncBN) and
    `mmcv.ops.sync_bn.SyncBatchNorm`(MMSyncBN) layers in the model to
    `BatchNormXd` layers.

    Adapted from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)

    Args:
        module (nn.Module): The module containing `SyncBatchNorm` layers.

    Returns:
        module_output: The converted module with `BatchNormXd` layers.
    """
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]

    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(module.num_features, module.eps, module.momentum, module.affine,
                                     module.track_running_stats)
        if module.affine:
            # no_grad() may not be needed here but
            # just to be consistent with `convert_sync_batchnorm()`
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        # qconfig exists in quantized models
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        # Some custom modules or 3rd party implemented modules may raise an
        # error when calling `add_module`. Therefore, try to catch the error
        # and do not raise it. See https://github.com/open-mmlab/mmengine/issues/638 # noqa: E501
        # for more details.
        try:
            module_output.add_module(name, revert_sync_batchnorm(child))
        except Exception:
            print(f'Failed to convert {child} from SyncBN to BN!')
    del module
    return module_output
