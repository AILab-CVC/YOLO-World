## Fine-tune YOLO-World on MS-COCO



### COCO Fine-tuning

##### NOTE:

1. Fine-tune models **without** `mask-refine` have some unknow errors and are under evaluation.
2. `X` and `S` models are coming soon.

| model | `mask-refine`| efficient neck |  AP | AP<sub>50</sub> | AP<sub>75</sub> | weights | log |
| :---- | :----------: |:-------------: | :-: | :--------------:| :-------------: |:------: | :-: |
| YOLO-World-v2-S | ✔️  | ✖️ | 45.7 | 62.0 | 49.9 | [HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-e6c2261e.pth) | [log](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_20240119_121515.log) |
| YOLO-World-v2-M | ✔️  | ✖️ | 50.7 | 67.5 | 55.2 | [HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-c6232481.pth) | [log](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_20240320_204957.log) |
| YOLO-World-v2-L | ✔️  | ✖️ | 53.3 | 70.3 | 58.0 | [HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-ac9177d6.pth) | [log](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_20240317_171126.log) |
| YOLO-World-v2-X | ✔️  | ✖️ | - | - | - | - | - |


