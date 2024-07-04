# Copyright (c) OpenMMLab. All rights reserved.
# Apache-2.0 license
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Number
from typing import List, Mapping, Optional, Sequence, Union

__all__ = ('YOLOWDetDataPreprocessor', )


class YOLOWDetDataPreprocessor(nn.Module):
    """Image pre-processor for detection tasks.
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = True):
        super().__init__()
        self._non_blocking = non_blocking

        assert not (bgr_to_rgb and rgb_to_bgr), ('`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        assert (mean is None) == (std is None), ('mean and std should be both None or tuple')

        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, ('`mean` should have 1 or 3 values, to be compatible with '
                                                      f'RGB or gray image, but got {len(mean)} values')
            assert len(std) == 3 or len(std) == 1, (  # type: ignore
                '`std` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std)} values')  # type: ignore
            self._enable_normalize = True
            self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def cast_data(self, data: dict) -> dict:
        """Copying data to the target device.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, (list, tuple)) and hasattr(data, '_fields'):
            # namedtuple
            return type(data)(*(self.cast_data(sample) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, torch.Tensor):
            return data.to(self.mean.device, non_blocking=self._non_blocking)
        else:
            return data

    def _samplelist_boxtype2tensor(self, batch_data_samples) -> list:
        # TODO: to be removed when dataset is reimplemented
        new_batch_data_samples = []
        for data_samples in batch_data_samples:
            new_data_samples = {'img_metas': {}}
            for k, v in data_samples['img_metas'].items():
                # ['texts', 'ori_shape', 'img_id', 'img_path', 'scale_factor', 'img_shape', 'pad_param']
                new_data_samples['img_metas'][k] = v
                new_data_samples[k] = v  # TODO removed, for debug
            if 'gt_instances' in data_samples:
                new_data_samples['gt_instances'] = data_samples['gt_instances']
            if 'pred_instances' in data_samples:
                new_data_samples['pred_instances'] = data_samples['pred_instances']
            if 'ignored_instances' in data_samples:
                new_data_samples['ignored_instances'] = data_samples['ignored_instances']
            new_batch_data_samples.append(new_data_samples)
        return new_batch_data_samples

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion
        """
        inputs, data_samples = data['inputs'], data['data_samples']

        if not training:
            assert len(inputs) == 1, 'only support batch_size=1 for test'
            inputs = torch.stack(inputs)
            data_samples = self._samplelist_boxtype2tensor(data_samples)

        batch_pad_shape = self._get_pad_shape(inputs)
        inputs = self.cast_data(inputs)
        data_samples = self.cast_data(data_samples)

        assert isinstance(inputs, torch.Tensor)
        assert isinstance(data_samples, dict) or \
            isinstance(data_samples, list)

        # TODO: Supports multi-scale training
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        # not used here
        h, w = inputs.shape[2:]
        target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
        target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
        pad_h = target_h - h
        pad_w = target_w - w
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h), 'constant', self.pad_value)

        assert tuple(inputs[0].size()[-2:]) == tuple(inputs.shape[2:])  # debug

        if not training:
            for idx, pad_shape in enumerate(batch_pad_shape):
                data_samples[idx]['img_metas']['batch_input_shape'] = tuple(inputs.shape[2:])
                data_samples[idx]['img_metas']['pad_shape'] = pad_shape
            return {'inputs': inputs, 'data_samples': data_samples}

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'texts': data_samples['texts'],
            'img_metas': img_metas
        }
        if 'masks' in data_samples:
            data_samples_output['masks'] = data_samples['masks']
        if 'is_detection' in data_samples:
            data_samples_output['is_detection'] = data_samples['is_detection']

        return {'inputs': inputs, 'data_samples': data_samples_output}

    def _get_pad_shape(self, _batch_inputs: torch.Tensor) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        assert _batch_inputs.dim() == 4, ('The input of `DataPreprocessor` should be a NCHW tensor '
                                          'or a list of tensor, but got a tensor with shape: '
                                          f'{_batch_inputs.shape}')
        pad_h = int(np.ceil(_batch_inputs.shape[2] / self.pad_size_divisor)) * self.pad_size_divisor
        pad_w = int(np.ceil(_batch_inputs.shape[3] / self.pad_size_divisor)) * self.pad_size_divisor
        batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        return batch_pad_shape
