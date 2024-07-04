# Copyright (c) Tencent Inc. All rights reserved.
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Union

from .datasets import MultiModalDataset, YOLOv5LVISV1Dataset
from .misc import pseudo_collate
from .samplers import DefaultSampler
from .transforms import YOLOResize, LoadAnnotations, LoadImageFromFile, LoadText, PackDetInputs

__all__ = ('build_lvis_testloader', )


def build_lvis_testloader(img_scale: Union[int, Tuple[int, int]] = (640, 640),
                          val_batch_size_per_gpu: int = 1,
                          val_num_workers: int = 2,
                          persistent_workers: bool = True,
                          seed: Optional[int] = None,
                          diff_rank_seed: bool = False) -> DataLoader:

    # build transform
    test_pipeline = [
        LoadImageFromFile(),
        YOLOResize(scale=img_scale),
        LoadAnnotations(with_bbox=True),
        LoadText(),
        PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape',
                                 'img_shape', 'scale_factor', 'pad_param', 'texts')),
    ]

    # build dataset
    lvis_dataset = YOLOv5LVISV1Dataset(
        data_root='data/coco/',
        test_mode=True,
        ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img=''),
        # batch_shapes_cfg=None
    )
    test_dataset = MultiModalDataset(
        dataset=lvis_dataset, class_text_path='data/texts/lvis_v1_class_texts.json', pipeline=test_pipeline)
    if hasattr(test_dataset, 'full_init'):
        test_dataset.full_init()

    # build sampler
    test_sampler = DefaultSampler(dataset=test_dataset, shuffle=False, seed=None if diff_rank_seed else seed)

    # build dataloader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=test_sampler,
        batch_size=val_batch_size_per_gpu,
        num_workers=val_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=pseudo_collate  # `pseudo_collate`
    )

    return test_dataloader
