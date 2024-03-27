## Fine-tune YOLO-World on MS-COCO


### Updates

1. [2024-3-27]: Considering that fine-tuning YOLO-World on COCO **without `mask-refine`** obtains bad results, e.g., YOLO-World-L obtains 48.6 AP without `mask-refine` compared to 53.3 AP with `mask-refine`, we rethink the training process and explore new training schemes for fine-tuning without `mask-refine`.
BTW, the COCO fine-tuning results are updated with higher performance (with `mask-refine`)!


### COCO Results and Checkpoints

**NOTE:**
1. AP<sup>ZS</sup>: AP evaluated in the zero-shot setting (w/o fine-tuning on COCO dataset).
2. `mask-refine`: refine the box annotations with masks, and add `CopyPaste` augmentation during training.

| model | Schedule | `mask-refine` | efficient neck | AP<sup>ZS</sup>|  AP | AP<sub>50</sub> | AP<sub>75</sub> | weights | log |
| :---- | :-------: | :----------: |:-------------: | :------------: | :-: | :--------------:| :-------------: |:------: | :-: |
| [YOLO-World-v2-S](./yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py) | AdamW, 2e-4, 80e | ✔️  | ✖️ | 37.5 | 45.7 | 62.0 | 49.9 | [HF Checkpoints]() | [log]() |
| [YOLO-World-v2-M](./yolo_world_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py) | AdamW, 2e-4, 80e | ✔️  | ✖️ | 42.8 | 51.0 | 67.5 | 55.2 | [HF Checkpoints]() | [log]() |
| [YOLO-World-v2-L](./yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py) | AdamW, 2e-4, 80e | ✔️  | ✖️ | 45.1 | 53.9 | 70.9 | 58.8 | [HF Checkpoints]() | [log]() |
| [YOLO-World-v2-L](./yolo_world_v2_l_efficient_neck_2e-4_80e_8gpus_mask-refine_finetune_coco.py) | AdamW, 2e-4, 80e | ✔️  |  ✔️ | 45.1 | |  | | [HF Checkpoints]() | [log]() |
| [YOLO-World-v2-X](./yolo_world_v2_x_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py) | AdamW, 2e-4, 80e | ✔️  | ✖️ | 46.8 | 54.7 | 71.6 | 59.6 | [HF Checkpoints]() | [log]() |
| [YOLO-World-v2-L]() | SGD, 1e-3, 40e | ✖️  | ✖️ | 45.1 |  |  |  | [HF Checkpoints]() | [log]() |



