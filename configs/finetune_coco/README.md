## Fine-tune YOLO-World on MS-COCO



### COCO Fine-tuning

##### NOTE:

1. AP<sup>ZS</sup>: AP evaluated in the zero-shot setting (w/o fine-tuning on COCO dataset).
2. Fine-tune models **without** `mask-refine` have some unknow errors and are under evaluation.
2. `X` models are coming soon.


| model | `mask-refine`| efficient neck | AP<sup>ZS</sup>|  AP | AP<sub>50</sub> | AP<sub>75</sub> | weights | log |
| :---- | :----------: |:-------------: | :------------: | :-: | :--------------:| :-------------: |:------: | :-: |
| [YOLO-World-v2-S](./yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py) | ✔️  | ✖️ | 37.5 | 45.7 | 62.0 | 49.9 | [HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-e6c2261e.pth) | [log](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_20240119_121515.log) |
| [YOLO-World-v2-M](yolo_world_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py) | ✔️  | ✖️ | 42.8 |50.7 | 67.5 | 55.2 | [HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-c6232481.pth) | [log](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_20240320_204957.log) |
| [YOLO-World-v2-L](./yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py) | ✔️  | ✖️ | 45.1 | 53.3 | 70.3 | 58.0 | [HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-ac9177d6.pth) | [log](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_20240317_171126.log) |
| [YOLO-World-v2-L](./yolo_world_v2_l_efficient_neck_2e-4_80e_8gpus_mask-refine_finetune_coco.py) | ✔️  |  ✔️ | 45.1 | |  | | [HF Checkpoints]() | [log]() |
| [ YOLO-World-v2-X](./yolo_world_v2_x_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py) | ✔️  | ✖️ | 46.8 | | - | - | [HF Checkpoints]() | [log]() |


