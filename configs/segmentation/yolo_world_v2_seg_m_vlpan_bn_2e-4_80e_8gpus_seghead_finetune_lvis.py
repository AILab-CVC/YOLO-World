_base_ = (
    '../../third_party/mmyolo/configs/yolov8/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco.py'
)
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)
# hyper-parameters
num_classes = 1203
num_training_classes = 80
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4

weight_decay = 0.05
train_batch_size_per_gpu = 8
load_from = 'pretrained_models/yolo_world_m_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-c6237d5b.pth'
text_model_name = '../pretrained_models/clip-vit-base-patch32-projection'
# text_model_name = 'openai/clip-vit-base-patch32'
persistent_workers = False

# Polygon2Mask
downsample_ratio = 4
mask_overlap = False
use_mask2refine = True
max_aspect_ratio = 100
min_area_ratio = 0.01

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        frozen_stages=4,  # frozen the image backbone
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              freeze_all=True,
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldSegHead',
                   head_module=dict(type='YOLOWorldSegHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes,
                                    mask_channels=32,
                                    proto_channels=256,
                                    freeze_bbox=True),
                   mask_overlap=mask_overlap,
                   loss_mask=dict(type='mmdet.CrossEntropyLoss',
                                  use_sigmoid=True,
                                  reduction='none'),
                   loss_mask_weight=1.0),
    train_cfg=dict(assigner=dict(
        type='YOLOWorldSegAssigner',
        num_classes=num_training_classes)),
    test_cfg=dict(mask_thr_binary=0.5, fast_test=True))

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=True,
         mask2bbox=True)
]

last_transform = [
    dict(type='mmdet.Albu',
         transforms=_base_.albu_train_transforms,
         bbox_params=dict(type='BboxParams',
                          format='pascal_voc',
                          label_fields=['gt_bboxes_labels',
                                        'gt_ignore_flags']),
         keymap={
             'img': 'image',
             'gt_bboxes': 'bboxes'
         }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='Polygon2Mask',
         downsample_ratio=downsample_ratio,
         mask_overlap=mask_overlap),
]

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
mosaic_affine_transform = [
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=pre_transform),
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=True)
]
train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(type='YOLOv5MultiModalMixUp',
         prob=_base_.mixup_prob,
         pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform, *text_transform
]

_train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(type='LetterResize',
         scale=_base_.img_scale,
         allow_scale_up=True,
         pad_val=dict(img=114.0)),
    dict(type='YOLOv5RandomAffine',
         max_rotate_degree=0.0,
         max_shear_degree=0.0,
         scaling_ratio_range=(1 - _base_.affine_scale,
                              1 + _base_.affine_scale),
         max_aspect_ratio=_base_.max_aspect_ratio,
         border_val=(114, 114, 114),
         min_area_ratio=min_area_ratio,
         use_mask_refine=use_mask2refine), *last_transform
]
train_pipeline_stage2 = [*_train_pipeline_stage2, *text_transform]
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='data/coco',
                 ann_file='lvis/lvis_v1_train_base.json',
                 data_prefix=dict(img=''),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32)),
    class_text_path='data/texts/lvis_v1_base_class_texts.json',
    pipeline=train_pipeline)
train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

# training settings
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0),
                                            'neck':
                                            dict(lr_mult=0.0),
                                            'head.head_module.reg_preds':
                                            dict(lr_mult=0.0),
                                            'head.head_module.cls_preds':
                                            dict(lr_mult=0.0),
                                            'head.head_module.cls_contrasts':
                                            dict(lr_mult=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='data/coco/',
                 test_mode=True,
                 ann_file='lvis/lvis_v1_val.json',
                 data_prefix=dict(img=''),
                 batch_shapes_cfg=None),
    class_text_path='data/texts/lvis_v1_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='data/coco/lvis/lvis_v1_val.json',
                     metric=['bbox', 'segm'])
test_evaluator = val_evaluator
find_unused_parameters = True
