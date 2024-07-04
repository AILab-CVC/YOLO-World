# Modify from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/base_det_dataset.py
# Apache-2.0 license
import copy
import gc
import json
import numpy as np
import os.path as osp
import pickle
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import Callable, List, Optional, Tuple, Union

__all__ = ('BaseDataset')


class BaseDataset(Dataset):

    METAINFO: dict = dict()
    _fully_initialized: bool = False

    def __init__(self,
                 data_root: Optional[str] = '',
                 ann_file: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 pipeline: List[Callable] = [],
                 test_mode: bool = False) -> None:

        self.data_root = data_root
        self.ann_file = ann_file
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        self.seg_map_suffix = '.png'
        self.return_classes = False
        self.max_refetch = 1000

        self._metainfo = copy.deepcopy(self.METAINFO)
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        self._join_prefix()
        self.full_init()

    def _join_prefix(self):
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file and not osp.isabs(self.ann_file) and self.data_root:
            self.ann_file = osp.join(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError('prefix should be a string, but got '
                                f'{type(prefix)}')
            if not osp.isabs(prefix) and self.data_root:
                self.data_prefix[data_key] = osp.join(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    def full_init(self):
        if self._fully_initialized: return
        # load data information
        self.data_list = self.load_data_list()
        self.data_list = self.filter_data()
        self.data_bytes, self.data_address = self._serialize_data()
        self._fully_initialized = True

    def filter_data(self) -> List[dict]:
        # by default
        return self.data_list

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def load_data_list(self) -> List[dict]:
        with open(self.ann_file, 'r') as jf:
            annotations = json.load(jf)
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            data_info = self.parse_data_info(raw_data_info)
            data_list.append(data_info)
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        # parse raw data information to target format
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (f'raw_data_info: {raw_data_info} dose not contain prefix key'
                                                 f'{prefix_key}, please check your data_prefix.')
            raw_data_info[prefix_key] = osp.join(prefix, raw_data_info[prefix_key])
        return raw_data_info

    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize ``self.data_list`` to save memory when launching multiple
        workers in data loading. This function will be called in ``full_init``.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        # Serialize data information list avoid making multiple copies of
        # `self.data_list` when iterate `import torch.utils.data.dataloader`
        # with multiple workers.
        data_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        # TODO Check if np.concatenate is necessary
        data_bytes = np.concatenate(data_list)
        # Empty cache for preventing making multiple copies of
        # `self.data_info` when loading data multi-processes.
        self.data_list.clear()
        gc.collect()
        return data_bytes, data_address

    def get_data_info(self, idx: int) -> dict:
        # if not self._fully_initialized:
        #     self.full_init()
        assert self._fully_initialized, \
            'The dataset has not been initialized.'
        start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
        end_addr = self.data_address[idx].item()
        bytes = memoryview(self.data_bytes[start_addr:end_addr])  # type: ignore
        data_info = pickle.loads(bytes)  # type: ignore
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info

    def prepare_data(self, idx: int):
        """Pass the dataset to the pipeline during training to support mixed
        data augmentation, such as Mosaic and MixUp."""
        data_info = self.get_data_info(idx)
        if self.test_mode is False:
            data_info['dataset'] = self
        return self.pipeline(data_info)

    def __len__(self) -> int:
        assert self._fully_initialized, \
            'The dataset has not been initialized.'
        return len(self.data_address)

    def __getitem__(self, idx: int) -> dict:
        if not self._fully_initialized:
            print('Please call `full_init()` method manually to accelerate '
                  'the speed.')
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = np.random.randint(0, len(self))
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')
