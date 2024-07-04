# Copyright (c) Tencent Inc. All rights reserved.
import copy
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import Callable, List

__all__ = ('MultiModalDataset', 'MultiModalMixedDataset')


class MultiModalDataset:
    """Multi-modal dataset."""

    def __init__(self,
                 dataset: Dataset,
                 class_text_path: str = None,
                 test_mode: bool = True,
                 pipeline: List[Callable] = [],
                 lazy_init: bool = False) -> None:
        self.dataset = dataset

        if class_text_path is not None:
            with open(class_text_path, 'r') as jf:
                self.class_texts = json.load(jf)
            # ori_classes = self.dataset.metainfo['classes']
            # assert len(ori_classes) == len(self.class_texts), \
            #     ('The number of classes in the dataset and the class text'
            #      'file must be the same.')
        else:
            self.class_texts = None

        self.test_mode = test_mode
        self._metainfo = self.dataset.metainfo
        self.pipeline = Compose(pipeline)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def full_init(self) -> None:
        """``full_init`` dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        assert self._fully_initialized, \
            'The dataset has not been initialized.'
        data_info = self.dataset.get_data_info(idx)
        if self.class_texts is not None:
            data_info.update({'texts': self.class_texts})
        return data_info

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print('Please call `full_init` method manually to '
                  'accelerate the speed.')
            self.full_init()

        data_info = self.get_data_info(idx)

        if hasattr(self.dataset, 'test_mode') and not self.dataset.test_mode:
            data_info['dataset'] = self
        elif not self.test_mode:
            data_info['dataset'] = self
        return self.pipeline(data_info)

    def __len__(self) -> int:
        assert self._fully_initialized, \
            'The dataset has not been initialized.'
        return self._ori_len


class MultiModalMixedDataset(MultiModalDataset):
    """Multi-modal Mixed dataset.
    mix "detection dataset" and "caption dataset"
    Args:
        dataset_type (str): dataset type, 'detection' or 'caption'
    """

    def __init__(self,
                 dataset: Dataset,
                 class_text_path: str = None,
                 dataset_type: str = 'detection',
                 test_mode: bool = True,
                 pipeline: List[Callable] = [],
                 lazy_init: bool = False) -> None:
        self.dataset_type = dataset_type
        super().__init__(dataset, class_text_path, test_mode, pipeline, lazy_init)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = self.dataset.get_data_info(idx)
        if self.class_texts is not None:
            data_info.update({'texts': self.class_texts})
        data_info['is_detection'] = 1 \
            if self.dataset_type == 'detection' else 0
        return data_info
