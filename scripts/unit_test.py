# -*- coding: utf-8 -*-
"""
Unittest.
"""

import logging
import torch
import unittest

from yolow.data import build_lvis_testloader
from yolow.engine.eval import LVISMetric
from yolow.logger import setup_logger
from yolow.model import (build_yolov8_backbone, build_yoloworld_backbone, build_yoloworld_data_preprocessor,
                         build_yoloworld_detector, build_yoloworld_head, build_yoloworld_neck, build_yoloworld_text)


class YOLOWUnitTest(unittest.TestCase):

    model_size = 's'
    # manual args, eg.
    model_args = dict()
    model_ckpt = './pretrained_weights/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'

    def setUp(self):
        setup_logger(name='unit test')

    def test_model(self):
        logger = logging.getLogger('unit test')

        # test build model
        logger.info('building data_preprocessor')
        data_preprocessor = build_yoloworld_data_preprocessor(self.model_size, args=self.model_args)
        logger.info('building yolov8_backbone')
        yolov8_backbone = build_yolov8_backbone(self.model_size, args=self.model_args)
        logger.info('building text_backbone')
        text_backbone = build_yoloworld_text(self.model_size, args=self.model_args)
        logger.info('building yolow_backbone')
        yolow_backbone = build_yoloworld_backbone(self.model_size, yolov8_backbone, text_backbone, args=self.model_args)
        logger.info('building yolow_neck')
        yolow_neck = build_yoloworld_neck(self.model_size, args=self.model_args)
        logger.info('building yolow_head')
        yolow_head = build_yoloworld_head(self.model_size, args=self.model_args)
        logger.info('building yoloworld_model')
        yoloworld_model = build_yoloworld_detector(
            self.model_size, yolow_backbone, yolow_neck, yolow_head, data_preprocessor, args=self.model_args)
        logger.info(f'yoloworld architecture: {yoloworld_model}')

        # test load ckpt (optional)
        logger.info('loading pretrained checkpoint')
        ckpt = torch.load(self.model_ckpt)
        yoloworld_model.load_state_dict(ckpt['state_dict'], strict=True)  # return (missing_keys, unexpected_keys)

        logger.info('yolow.model is OK.')

    def test_data(self):
        logger = logging.getLogger('unit test')

        # test build lvis test_dataloader
        logger.info('building LVIS test_loader')
        loader = build_lvis_testloader()
        logger.info(f'LVIS test_loader: {loader}')

        logger.info('yolow.data is OK.')

    def test_evaluator(self):
        logger = logging.getLogger('unit test')

        # test build lvis eval_metric
        logger.info('building LVIS evaluator')
        evaluator = LVISMetric(ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json', metric='bbox')
        logger.info(f'LVIS evaluator: {evaluator}')

        logger.info('yolow.engine.eval is OK.')


if __name__ == '__main__':
    unittest.main()
