# Copyright (c) Tencent Inc. All rights reserved.
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Sequence, Tuple, Union

from .layers import Conv
from .losses import CIoULoss, CrossEntropyLoss, DistributionFocalLoss
from .misc import (batched_nms, filter_scores_and_topk, get_box_wh, get_dist_info, gt_instances_preprocess,
                   make_divisible, multi_apply, scale_boxes, unpack_gt_instances, yolow_dict)
from .task_utils import DistancePointBBoxCoder, MlvlPointGenerator

__all__ = (
    'YOLOWorldHeadModule',
    'YOLOWorldHead',
)


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World
    """

    def __init__(self, use_einsum: bool = True) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


class BNContrastiveHead(nn.Module):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    """

    def __init__(self, embed_dims: int, use_einsum: bool = True) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims, momentum=0.03, eps=0.001)
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


class YOLOWorldHeadModule(nn.Module):
    """Head Module for YOLO-World
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 embed_dims: int,
                 use_bn_head: bool = True,
                 use_einsum: bool = True,
                 freeze_all: bool = False,
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 with_norm: bool = True,
                 with_activation: bool = True) -> None:
        super().__init__()

        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.with_norm = with_norm
        self.with_activation = with_activation
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        for reg_pred, cls_pred, cls_contrast, stride in zip(self.reg_preds, self.cls_preds, self.cls_contrasts,
                                                            self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(cls_contrast.bias.data, math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    Conv(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        with_norm=self.with_norm,
                        with_activation=self.with_activation),
                    Conv(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        with_norm=self.with_norm,
                        with_activation=self.with_activation),
                    nn.Conv2d(in_channels=reg_out_channels, out_channels=4 * self.reg_max, kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    Conv(
                        in_channels=self.in_channels[i],
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        with_norm=self.with_norm,
                        with_activation=self.with_activation),
                    Conv(
                        in_channels=cls_out_channels,
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        with_norm=self.with_norm,
                        with_activation=self.with_activation),
                    nn.Conv2d(in_channels=cls_out_channels, out_channels=self.embed_dims, kernel_size=1)))
            if self.use_bn_head:
                self.cls_contrasts.append(BNContrastiveHead(self.embed_dims, use_einsum=self.use_einsum))
            else:
                self.cls_contrasts.append(ContrastiveHead(self.embed_dims, use_einsum=self.use_einsum))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        return multi_apply(self.forward_single, img_feats, txt_feats, self.cls_preds, self.reg_preds,
                           self.cls_contrasts)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor, cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


class YOLOWorldHead(nn.Module):
    """YOLO-World Head

    - loss(): forward() -> loss_by_feat()
    - predict(): forward() -> predict_by_feat()
    - loss_and_predict(): forward() -> loss_by_feat() -> predict_by_feat()
    """

    def __init__(self, head_module: nn.Module, test_cfg: Optional[dict] = None) -> None:
        super().__init__()

        self.head_module = head_module
        self.num_classes = self.head_module.num_classes
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)

        self.test_cfg = test_cfg

        # init losses
        self.loss_cls = CrossEntropyLoss(
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5,
        )
        self.loss_bbox = CIoULoss(
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            eps=1e-7,
        )
        self.loss_dfl = DistributionFocalLoss(
            reduction='mean',
            loss_weight=1.5 / 4,
        )

        # init task_utils
        self.prior_generator = MlvlPointGenerator(offset=0.5, strides=[8, 16, 32])
        self.bbox_coder = DistancePointBBoxCoder()
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.featmap_sizes = [torch.empty(1)] * self.num_levels

        self.prior_match_thr = 4.0
        self.near_neighbor_thr = 0.5
        self.obj_level_weights = [4.0, 1.0, 0.4]
        self.ignore_iof_thr = -1.0

        # fixed train_cfg for ease
        # TODO later
        # self.assigner = BatchTaskAlignedAssigner(
        #     num_classes=self.num_classes,
        #     use_ciou=True,  # only support ciou
        #     topk=10,  # Number of bbox selected in each level
        #     alpha=0.5,  # A Hyper-parameter related to alignment_metrics
        #     beta=6.0,  # A Hyper-parameter related to alignment_metrics
        #     eps=1e-9)

        # Add common attributes to reduce calculation
        self.featmap_sizes_train = None
        self.num_level_priors = None
        self.flatten_priors_train = None
        self.stride_tensor = None

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor) -> Tuple[List]:
        return self.head_module(img_feats, txt_feats)

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor, batch_data_samples: Union[list, dict]) -> dict:
        outs = self(img_feats, txt_feats)
        # Fast version
        loss_inputs = outs + (batch_data_samples['bboxes_labels'], batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_and_predict(self,
                         img_feats: Tuple[Tensor],
                         txt_feats: Tensor,
                         batch_data_samples: Union[list, dict],
                         proposal_cfg: Optional[dict] = None) -> Tuple[dict, list]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs

        outs = self(img_feats, txt_feats)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)
        return losses, predictions

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: Union[list, dict],
                rescale: bool = False) -> list:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        batch_img_metas = [
            data_samples['img_metas'] for data_samples in batch_data_samples  # changed `.metainfo` to ['img_metas']
        ]
        outs = self(img_feats, txt_feats)
        predictions = self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def loss_by_feat(self,
                     cls_scores: Sequence[Tensor],
                     bbox_preds: Sequence[Tensor],
                     bbox_dist_preds: Sequence[Tensor],
                     batch_gt_instances: Sequence[dict],
                     batch_img_metas: Sequence[dict],
                     batch_gt_instances_ignore: Optional[list] = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4) for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(self.flatten_priors_train[..., :2], flatten_pred_bboxes,
                                                     self.stride_tensor[..., 0])

        # TODO later
        # assigned_result = self.assigner((flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
        #                                 flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train, gt_labels,
        #                                 gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(pred_bboxes_pos, assigned_bboxes_pos, weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        # if self.world_size == -1:
        _, world_size = get_dist_info()
        # else:
        # world_size = self.world_size
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[dict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List:
        """Transform a batch of output features extracted by the head into
        bbox results.
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full((featmap_size.numel() * self.num_base_priors, ), stride)
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]
        # 8400
        # print(flatten_cls_scores.shape)
        results_list = []
        for (bboxes, scores, objectness, img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                                                          flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get('yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = yolow_dict()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores, score_thr, nms_pre, results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(scores, score_thr, nms_pre)

            results = yolow_dict(scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor(
                        [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
                results.bboxes /= results.bboxes.new_tensor(scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results, cfg=cfg, rescale=False, with_nms=with_nms, img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list

    def _bbox_post_process(self,
                           results: dict,
                           cfg: dict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> dict:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO: Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            # bboxes = get_box_tensor(results.bboxes)
            bboxes = results.bboxes
            assert isinstance(bboxes, Tensor)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.labels, cfg.nms)
            # results = results[keep_idxs]
            for k in results.keys():
                results[k] = results[k][keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            # results = results[:cfg.max_per_img]
            for k in results.keys():
                results[k] = results[k][:cfg.max_per_img]

        return results

    # def _convert_gt_to_norm_format(self,
    #                                batch_gt_instances: Sequence[dict],
    #                                batch_img_metas: Sequence[dict]) -> Tensor:
    #     if isinstance(batch_gt_instances, torch.Tensor):
    #         # fast version
    #         img_shape = batch_img_metas[0]['batch_input_shape']
    #         gt_bboxes_xyxy = batch_gt_instances[:, 2:]
    #         xy1, xy2 = gt_bboxes_xyxy.split((2, 2), dim=-1)
    #         gt_bboxes_xywh = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
    #         gt_bboxes_xywh[:, 1::2] /= img_shape[0]
    #         gt_bboxes_xywh[:, 0::2] /= img_shape[1]
    #         batch_gt_instances[:, 2:] = gt_bboxes_xywh

    #         # (num_base_priors, num_bboxes, 6)
    #         batch_targets_normed = batch_gt_instances.repeat(
    #             self.num_base_priors, 1, 1)
    #     else:
    #         batch_target_list = []
    #         # Convert xyxy bbox to yolo format.
    #         for i, gt_instances in enumerate(batch_gt_instances):
    #             img_shape = batch_img_metas[i]['batch_input_shape']
    #             bboxes = gt_instances.bboxes
    #             labels = gt_instances.labels

    #             xy1, xy2 = bboxes.split((2, 2), dim=-1)
    #             bboxes = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
    #             # normalized to 0-1
    #             bboxes[:, 1::2] /= img_shape[0]
    #             bboxes[:, 0::2] /= img_shape[1]

    #             index = bboxes.new_full((len(bboxes), 1), i)
    #             # (batch_idx, label, normed_bbox)
    #             target = torch.cat((index, labels[:, None].float(), bboxes),
    #                                dim=1)
    #             batch_target_list.append(target)

    #         # (num_base_priors, num_bboxes, 6)
    #         batch_targets_normed = torch.cat(
    #             batch_target_list, dim=0).repeat(self.num_base_priors, 1, 1)

    #     # (num_base_priors, num_bboxes, 1)
    #     batch_targets_prior_inds = self.prior_inds.repeat(
    #         1, batch_targets_normed.shape[1])[..., None]
    #     # (num_base_priors, num_bboxes, 7)
    #     # (img_ind, labels, bbox_cx, bbox_cy, bbox_w, bbox_h, prior_ind)
    #     batch_targets_normed = torch.cat(
    #         (batch_targets_normed, batch_targets_prior_inds), 2)
    #     return batch_targets_normed

    # def _decode_bbox_to_xywh(self, bbox_pred, priors_base_sizes) -> Tensor:
    #     bbox_pred = bbox_pred.sigmoid()
    #     pred_xy = bbox_pred[:, :2] * 2 - 0.5
    #     pred_wh = (bbox_pred[:, 2:] * 2)**2 * priors_base_sizes
    #     decoded_bbox_pred = torch.cat((pred_xy, pred_wh), dim=-1)
    #     return decoded_bbox_pred

    # def _loss_by_feat_with_ignore(
    #         self, cls_scores: Sequence[Tensor], bbox_preds: Sequence[Tensor],
    #         objectnesses: Sequence[Tensor],
    #         batch_gt_instances: Sequence[dict],
    #         batch_img_metas: Sequence[dict],
    #         batch_gt_instances_ignore: Sequence[Tensor]) -> dict:
    #     """Calculate the loss based on the features extracted by the detection
    #     head.
    #     """
    #     # 1. Convert gt to norm format
    #     batch_targets_normed = self._convert_gt_to_norm_format(
    #         batch_gt_instances, batch_img_metas)

    #     featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
    #     if featmap_sizes != self.featmap_sizes:
    #         self.mlvl_priors = self.prior_generator.grid_priors(
    #             featmap_sizes,
    #             dtype=cls_scores[0].dtype,
    #             device=cls_scores[0].device)
    #         self.featmap_sizes = featmap_sizes

    #     device = cls_scores[0].device
    #     loss_cls = torch.zeros(1, device=device)
    #     loss_box = torch.zeros(1, device=device)
    #     scaled_factor = torch.ones(7, device=device)

    #     for i in range(self.num_levels):
    #         batch_size, _, h, w = bbox_preds[i].shape
    #         target_obj = torch.zeros_like(objectnesses[i])

    #         not_ignore_flags = bbox_preds[i].new_ones(batch_size,
    #                                                   self.num_base_priors, h,
    #                                                   w)

    #         ignore_overlaps = bbox_overlaps(self.mlvl_priors[i],
    #                                         batch_gt_instances_ignore[..., 2:],
    #                                         'iof')
    #         ignore_max_overlaps, ignore_max_ignore_index = ignore_overlaps.max(
    #             dim=1)

    #         batch_inds = batch_gt_instances_ignore[:,
    #                                                0][ignore_max_ignore_index]
    #         ignore_inds = (ignore_max_overlaps > self.ignore_iof_thr).nonzero(
    #             as_tuple=True)[0]
    #         batch_inds = batch_inds[ignore_inds].long()
    #         ignore_priors, ignore_grid_xs, ignore_grid_ys = get_prior_xy_info(
    #             ignore_inds, self.num_base_priors, self.featmap_sizes[i])
    #         not_ignore_flags[batch_inds, ignore_priors, ignore_grid_ys,
    #                          ignore_grid_xs] = 0

    #         # empty gt bboxes
    #         if batch_targets_normed.shape[1] == 0:
    #             loss_box += bbox_preds[i].sum() * 0
    #             loss_cls += cls_scores[i].sum() * 0
    #             continue

    #         priors_base_sizes_i = self.priors_base_sizes[i]
    #         # feature map scale whwh
    #         scaled_factor[2:6] = torch.tensor(
    #             bbox_preds[i].shape)[[3, 2, 3, 2]]
    #         # Scale batch_targets from range 0-1 to range 0-features_maps size.
    #         # (num_base_priors, num_bboxes, 7)
    #         batch_targets_scaled = batch_targets_normed * scaled_factor

    #         # 2. Shape match
    #         wh_ratio = batch_targets_scaled[...,
    #                                         4:6] / priors_base_sizes_i[:, None]
    #         match_inds = torch.max(
    #             wh_ratio, 1 / wh_ratio).max(2)[0] < self.prior_match_thr
    #         batch_targets_scaled = batch_targets_scaled[match_inds]

    #         # no gt bbox matches anchor
    #         if batch_targets_scaled.shape[0] == 0:
    #             loss_box += bbox_preds[i].sum() * 0
    #             loss_cls += cls_scores[i].sum() * 0
    #             continue

    #         # 3. Positive samples with additional neighbors

    #         # check the left, up, right, bottom sides of the
    #         # targets grid, and determine whether assigned
    #         # them as positive samples as well.
    #         batch_targets_cxcy = batch_targets_scaled[:, 2:4]
    #         grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
    #         left, up = ((batch_targets_cxcy % 1 < self.near_neighbor_thr) &
    #                     (batch_targets_cxcy > 1)).T
    #         right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) &
    #                          (grid_xy > 1)).T
    #         offset_inds = torch.stack(
    #             (torch.ones_like(left), left, up, right, bottom))

    #         batch_targets_scaled = batch_targets_scaled.repeat(
    #             (5, 1, 1))[offset_inds]
    #         retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1],
    #                                                    1)[offset_inds]

    #         # prepare pred results and positive sample indexes to
    #         # calculate class loss and bbox lo
    #         _chunk_targets = batch_targets_scaled.chunk(4, 1)
    #         img_class_inds, grid_xy, grid_wh, priors_inds = _chunk_targets
    #         priors_inds, (img_inds, class_inds) = priors_inds.long().view(
    #             -1), img_class_inds.long().T

    #         grid_xy_long = (grid_xy -
    #                         retained_offsets * self.near_neighbor_thr).long()
    #         grid_x_inds, grid_y_inds = grid_xy_long.T
    #         bboxes_targets = torch.cat((grid_xy - grid_xy_long, grid_wh), 1)

    #         # 4. Calculate loss
    #         # bbox loss
    #         retained_bbox_pred = bbox_preds[i].reshape(
    #             batch_size, self.num_base_priors, -1, h,
    #             w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]
    #         priors_base_sizes_i = priors_base_sizes_i[priors_inds]
    #         decoded_bbox_pred = self._decode_bbox_to_xywh(
    #             retained_bbox_pred, priors_base_sizes_i)

    #         not_ignore_weights = not_ignore_flags[img_inds, priors_inds,
    #                                               grid_y_inds, grid_x_inds]
    #         loss_box_i, iou = self.loss_bbox(
    #             decoded_bbox_pred,
    #             bboxes_targets,
    #             weight=not_ignore_weights,
    #             avg_factor=max(not_ignore_weights.sum(), 1))
    #         loss_box += loss_box_i

    #         # obj loss
    #         iou = iou.detach().clamp(0)
    #         target_obj[img_inds, priors_inds, grid_y_inds,
    #                    grid_x_inds] = iou.type(target_obj.dtype)

    #         # cls loss
    #         if self.num_classes > 1:
    #             pred_cls_scores = cls_scores[i].reshape(
    #                 batch_size, self.num_base_priors, -1, h,
    #                 w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

    #             target_class = torch.full_like(pred_cls_scores, 0.)
    #             target_class[range(batch_targets_scaled.shape[0]),
    #                          class_inds] = 1.
    #             loss_cls += self.loss_cls(
    #                 pred_cls_scores,
    #                 target_class,
    #                 weight=not_ignore_weights[:, None].repeat(
    #                     1, self.num_classes),
    #                 avg_factor=max(not_ignore_weights.sum(), 1))
    #         else:
    #             loss_cls += cls_scores[i].sum() * 0

    #     _, world_size = get_dist_info()
    #     return dict(
    #         loss_cls=loss_cls * batch_size * world_size,
    #         loss_bbox=loss_box * batch_size * world_size)
