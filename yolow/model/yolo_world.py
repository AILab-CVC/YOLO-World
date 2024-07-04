# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from collections import OrderedDict
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

from .misc import get_world_size

__all__ = ('YOLOWorldDetector', )


class YOLOWorldDetector(nn.Module):
    """YOLO-World arch

    train_step(): forward() -> loss() -> extract_feat()
    val_step(): forward() -> predict() -> extract_feat()
    """

    def __init__(self,
                 backbone: nn.Module,
                 neck: nn.Module,
                 bbox_head: nn.Module,
                 mm_neck: bool = False,
                 num_train_classes: int = 80,
                 num_test_classes: int = 80,
                 data_preprocessor: Optional[nn.Module] = None,
                 use_syncbn: bool = True) -> None:
        super().__init__()

        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes

        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head
        self.data_preprocessor = data_preprocessor

        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    def train_step(self, data: Union[dict, tuple, list], optim_wrapper) -> Dict[str, torch.Tensor]:
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):  # TODO optim_wrapper
            data = self.data_preprocessor(data, True)
            losses = self(**data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    @torch.no_grad()
    def val_step(self, data: Union[tuple, dict, list]) -> list:
        data = self.data_preprocessor(data, False)
        return self(**data, mode='predict')  # type: ignore

    @torch.no_grad()
    def test_step(self, data: Union[dict, tuple, list]) -> list:
        return self.val_step(data)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[Union[List, dict]] = None,
                mode: str = 'tensor') -> Union[dict, list, tuple, Tensor]:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif isinstance(loss_value, Union[List[torch.Tensor], Tuple[torch.Tensor]]):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore

    def loss(self, batch_inputs: Tensor, batch_data_samples: Union[List, dict]) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: Union[List, dict], rescale: bool = True) -> list:
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)

        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats, txt_feats, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: Optional[Union[List, dict]] = None) -> Tuple[List[Tensor]]:
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(self, batch_inputs: Tensor, batch_data_samples: Union[List, dict]) -> Tuple[Tuple[Tensor], Tensor]:
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples, dict) and 'texts' in batch_data_samples['img_metas']:
            texts = batch_data_samples['img_metas']['texts']
        elif isinstance(batch_data_samples, list) and ('texts' in batch_data_samples[0]['img_metas']):
            texts = [data_sample['img_metas']['texts'] for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats

    def add_pred_to_datasample(self, data_samples: List, results_list: List) -> List:
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample['pred_instances'] = pred_instances
        # samplelist_boxtype2tensor(data_samples)
        return data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)
