## Fine-tuning YOLO-World

Fine-tuning YOLO-World is easy and we provide the samples for COCO object detection as a simple guidance.


### Fine-tuning Requirements

Fine-tuning YOLO-World is cheap:

* it does not require 32 GPUs for multi-node distributed training. **8 GPUs or even 1 GPU** is enough.

* it does not require the long schedule, *e.g.,* 300 epochs or 500 epochs for training YOLOv5 or YOLOv8. **80 epochs or fewer** is enough considering that we provide the good pre-trained weights.

### Data Preparation

The fine-tuning dataset should have the similar format as the that of the pre-training dataset.
We suggest you refer to [`docs/data`](./data.md) for more details about how to build the datasets:

* if you fine-tune YOLO-World for close-set / custom vocabulary object detection, using `MultiModalDataset` with a `text json` is preferred.

* if you fine-tune YOLO-World for open-vocabulary detection with rich texts or grounding tasks, using `MixedGroundingDataset` is preferred.

### Hyper-parameters and Config

Please refer to the [config for fine-tuning YOLO-World-L on COCO](../configs/finetune_coco/yolo_world_l_dual_vlpan_2e-4_80e_8gpus_finetune_coco.py) for more details.

1. Basic config file:

If the fine-tuning dataset **contains mask annotations**:

```python
_base_ = ('../../third_party/mmyolo/configs/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')
```

If the fine-tuning dataset **doesn't contain mask annotations**:

```python
_base_ = ('../../third_party/mmyolo/configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py')
```

2. Training Schemes:

Reducing the epochs and adjusting the learning rate

```python
max_epochs = 80
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
close_mosaic_epochs=10

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])

```

3. Datasets:

```python
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)
```

#### Finetuning without RepVL-PAN or Text Encoder 🚀

For further efficiency and simplicity, we can fine-tune an efficient version of YOLO-World without RepVL-PAN and the text encoder.
The efficient version of YOLO-World has the similar architecture or layers with the orignial YOLOv8 but we provide the pre-trained weights on large-scale datasets.
The pre-trained YOLO-World has strong generalization capabilities and is more robust compared to YOLOv8 trained on the COCO dataset.

You can refer to the [config for Efficient YOLO-World](./../configs/finetune_coco/yolo_world_l_efficient_neck_2e-4_80e_8gpus_finetune_coco.py) for more details.

The efficient YOLO-World adopts `EfficientCSPLayerWithTwoConv` and the text encoder can be removed during inference or exporting models.

```python

model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='EfficientCSPLayerWithTwoConv')))

```

### Launch Fine-tuning!

It's easy:

```bash
./dist_train.sh <path/to/config> <NUM_GPUS> --amp
```
