## Prompt Tuning for YOLO-World

### NOTE:

This folder contains many experimental config files, which will be removed later!!

### Experimental Results

| Model | Config |  AP  | AP50 | AP75  | APS | APM | APL |
| :---- | :----: | :--: | :--: | :---: | :-: | :-: | :-: |
| YOLO-World-v2-L | Zero-shot | 45.7 | 61.6 | 49.8 | 29.9 | 50.0 | 60.8 |
| [YOLO-World-v2-L](./../configs/prompt_tuning_coco/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_prompt_tuning_coco.py) | Prompt tuning | 47.9 | 64.3 | 52.5 | 31.9 | 52.6 | 61.3 | 
