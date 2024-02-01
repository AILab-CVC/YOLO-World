# Copyright (c) Tencent Inc. All rights reserved.
import copy
import json
import os.path as osp
from typing import List

from mmengine.fileio import get_local_path

from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets import CocoDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmyolo.registry import DATASETS

v3det_ignore_list = [
    'a00013820/26_275_28143226914_ff3a247c53_c.jpg',
    'n03815615/12_1489_32968099046_be38fa580e_c.jpg',
    'n04550184/19_1480_2504784164_ffa3db8844_c.jpg',
    'a00008703/2_363_3576131784_dfac6fc6ce_c.jpg',
    'n02814533/28_2216_30224383848_a90697f1b3_c.jpg',
    'n12026476/29_186_15091304754_5c219872f7_c.jpg',
    'n01956764/12_2004_50133201066_72e0d9fea5_c.jpg',
    'n03785016/14_2642_518053131_d07abcb5da_c.jpg',
    'a00011156/33_250_4548479728_9ce5246596_c.jpg',
    'a00009461/19_152_2792869324_db95bebc84_c.jpg',
]

# # ugly code here
# with open(osp.join("data/v3det/cats.json"), 'r') as f:
#     _classes = json.load(f)['classes']


@DATASETS.register_module()
class V3DetDataset(CocoDataset):
    """Objects365 v1 dataset for detection."""

    METAINFO = {'classes': 'classes', 'palette': None}

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(self.ann_file,
                            backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

        # 'categories' list in objects365_train.json and objects365_val.json
        # is inconsistent, need sort list(or dict) before get cat_ids.
        cats = self.coco.cats
        sorted_cats = {i: cats[i] for i in sorted(cats)}
        self.coco.cats = sorted_cats
        categories = self.coco.dataset['categories']
        sorted_categories = sorted(categories, key=lambda i: i['id'])
        self.coco.dataset['categories'] = sorted_categories
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            file_name = osp.join(
                osp.split(osp.split(raw_img_info['file_name'])[0])[-1],
                osp.split(raw_img_info['file_name'])[-1])

            if file_name in v3det_ignore_list:
                continue

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

        return data_list


@DATASETS.register_module()
class YOLOv5V3DetDataset(BatchShapePolicyDataset, V3DetDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with Objects365V1Dataset.
    See `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
