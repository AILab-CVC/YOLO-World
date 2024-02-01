# Copyright (c) Tencent Inc. All rights reserved.
import json
import random
from typing import Tuple

import numpy as np
from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomLoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '') -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        num_classes = len(class_texts)
        if 'gt_labels' in results:
            gt_label_tag = 'gt_labels'
        elif 'gt_bboxes_labels' in results:
            gt_label_tag = 'gt_bboxes_labels'
        else:
            raise ValueError('No valid labels found in results.')
        positive_labels = set(results[gt_label_tag])

        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))

        num_neg_samples = min(
            min(num_classes, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels:
                candidate_neg_labels.append(idx)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['gt_bboxes']), dtype=bool)
        for idx, label in enumerate(results[gt_label_tag]):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results[gt_label_tag][idx] = label2ids[label]
        results['gt_bboxes'] = results['gt_bboxes'][gt_valid_mask]
        results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]

        if 'instances' in results:
            retaged_instances = []
            for idx, inst in enumerate(results['instances']):
                label = inst['bbox_label']
                if label in label2ids:
                    inst['bbox_label'] = label2ids[label]
                    retaged_instances.append(inst)
            results['instances'] = retaged_instances

        texts = []
        for label in sampled_labels:
            cls_caps = class_texts[label]
            assert len(cls_caps) > 0
            cap_id = random.randrange(len(cls_caps))
            sel_cls_cap = self.prompt_format.format(cls_caps[cap_id])
            texts.append(sel_cls_cap)

        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        results['texts'] = texts

        return results


@TRANSFORMS.register_module()
class LoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 multi_prompt_flag: str = '/') -> None:
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        texts = []
        for idx, cls_caps in enumerate(class_texts):
            assert len(cls_caps) > 0
            sel_cls_cap = cls_caps[0]
            sel_cls_cap = self.prompt_format.format(sel_cls_cap)
            texts.append(sel_cls_cap)

        results['texts'] = texts

        return results
