_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
num_classes = 80
num_training_classes = 80
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
load_from = '../FastDet/output_models/pretrain_yolow-v8_s_clipv2_frozen_te_noprompt_t2i_bn_2e-3adamw_scale_lr_wd_32xb16-100e_obj365v1_goldg_cc3mram250k_train_lviseval-e3592307_rep_conv.pth'
persistent_workers = False
mixup_prob = 0.15
copypaste_prob = 0.3

# model settings
model = dict(type='SimpleYOLOWorldDetector',
             mm_neck=True,
             num_train_classes=num_classes,
             num_test_classes=num_classes,
             reparameterized=True,
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           text_model=None,
                           image_model={{_base_.model.backbone}},
                           with_text_model=False),
             neck=dict(type='YOLOWorldPAFPN',
                       guide_channels=text_channels,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='EfficientCSPLayerWithTwoConv')),
             bbox_head=dict(head_module=dict(type='RepYOLOWorldHeadModule',
                                             embed_dims=text_channels,
                                             num_guide=num_classes,
                                             num_classes=num_classes)),
             train_cfg=dict(assigner=dict(num_classes=num_classes)))

# dataset settings
final_transform = [
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
]
mosaic_affine_transform = [
    dict(type='Mosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=copypaste_prob),
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
        use_mask_refine=_base_.use_mask2refine)
]
train_pipeline = [
    *_base_.pre_transform, *mosaic_affine_transform,
    dict(type='YOLOv5MixUp',
         prob=mixup_prob,
         pre_transform=[*_base_.pre_transform, *mosaic_affine_transform]),
    *_base_.last_transform[:-1], *final_transform
]

train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *final_transform]

coco_train_dataset = dict(type='YOLOv5CocoDataset',
                          data_root='data/coco',
                          ann_file='annotations/instances_train2017.json',
                          data_prefix=dict(img='train2017/'),
                          filter_cfg=dict(filter_empty_gt=False, min_size=32),
                          pipeline=train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]
coco_val_dataset = dict(type='YOLOv5CocoDataset',
                        data_root='data/coco',
                        ann_file='annotations/instances_val2017.json',
                        data_prefix=dict(img='val2017/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=test_pipeline)

val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader
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
                     constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(_delete_=True,
                     type='mmdet.CocoMetric',
                     proposal_nums=(100, 1, 10),
                     ann_file='data/coco/annotations/instances_val2017.json',
                     metric='bbox')
