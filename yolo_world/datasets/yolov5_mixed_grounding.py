# Copyright (c) Tencent Inc. All rights reserved.
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path, join_path
from mmengine.utils import is_abs
from mmdet.datasets.coco import CocoDataset
from mmyolo.registry import DATASETS
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset


@DATASETS.register_module()
class YOLOv5MixedGroundingDataset(BatchShapePolicyDataset, CocoDataset):
    """Mixed grounding dataset."""

    METAINFO = {
        'classes': ('object',),
        'palette': [(220, 20, 60)]}

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco
        # print(len(data_list))
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        img_path = None
        img_prefix = self.data_prefix.get('img', None)
        if isinstance(img_prefix, str):
            img_path = osp.join(img_prefix, img_info['file_name'])
        elif isinstance(img_prefix, (list, tuple)):
            for prefix in img_prefix:
                candidate_img_path = osp.join(prefix, img_info['file_name'])
                if osp.exists(candidate_img_path):
                    img_path = candidate_img_path
                    break
        assert img_path is not None, (
            f'Image path {img_info["file_name"]} not found in'
            f'{img_prefix}')
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = float(img_info['height'])
        data_info['width'] = float(img_info['width'])

        cat2id = {}
        texts = []
        for ann in ann_info:
            cat_name = ' '.join([img_info['caption'][t[0]:t[1]]
                                 for t in ann['tokens_positive']])
            if cat_name not in cat2id:
                cat2id[cat_name] = len(cat2id)
                texts.append([cat_name])
        data_info['texts'] = texts

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0,
                          min(x1 + w, float(img_info['width'])) - max(x1, 0))
            inter_h = max(0,
                          min(y1 + h, float(img_info['height'])) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox

            cat_name = ' '.join([img_info['caption'][t[0]:t[1]]
                                 for t in ann['tokens_positive']])
            instance['bbox_label'] = cat2id[cat_name]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        # NOTE: for detection task, we set `is_detection` to 1
        data_info['is_detection'] = 1
        data_info['instances'] = instances
        # print(data_info['texts'])
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = int(data_info['width'])
            height = int(data_info['height'])
            if filter_empty_gt and img_id not in ids_with_ann:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.
        """
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file and not is_abs(self.ann_file) and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if isinstance(prefix, (list, tuple)):
                abs_prefix = []
                for p in prefix:
                    if not is_abs(p) and self.data_root:
                        abs_prefix.append(join_path(self.data_root, p))
                    else:
                        abs_prefix.append(p)
                self.data_prefix[data_key] = abs_prefix
            elif isinstance(prefix, str):
                if not is_abs(prefix) and self.data_root:
                    self.data_prefix[data_key] = join_path(
                        self.data_root, prefix)
                else:
                    self.data_prefix[data_key] = prefix
            else:
                raise TypeError('prefix should be a string, tuple or list,'
                                f'but got {type(prefix)}')
