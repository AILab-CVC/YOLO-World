## Fine-tuning YOLO-World for Instance Segmentation


### Models

We fine-tune YOLO-World on LVIS (`LVIS-Base`) with mask annotations for open-vocabulary (zero-shot) instance segmentation.

We provide two fine-tuning strategies YOLO-World towards open-vocabulary instance segmentation:

* fine-tuning `all modules`: leads to better LVIS segmentation accuracy but affects the zero-shot performance.

* fine-tuning the `segmentation head`: maintains the zero-shot performanc but lowers LVIS segmentation accuracy. 

| Model | Fine-tuning Data | Fine-tuning Modules| AP<sup>mask</su> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | Weights |
| :---- | :--------------- | :----------------: | :--------------: | :------------: | :------------: | :------------: | :-----: |
| [YOLO-World-Seg-M](./yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis.py) | `LVIS-Base` | `all modules` | 25.9 | 13.4 | 24.9 | 32.6  | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis-ca465825.pth) |
| [YOLO-World-v2-Seg-M](./yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis.py) | `LVIS-Base` | `all modules` | 25.9 | 13.4 | 24.9 | 32.6  | [HF Checkpoints 🤗]() |
| [YOLO-World-Seg-L](./yolo_world_seg_l_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis.py) | `LVIS-Base` | `all modules` | 28.7 | 15.0 | 28.3 | 35.2| [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_seg_l_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis-8c58c916.pth) |
| [YOLO-World-v2-Seg-L](./yolo_world_seg_l_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis.py) | `LVIS-Base` | `all modules` | 28.7 | 15.0 | 28.3 | 35.2| [HF Checkpoints 🤗]() |
| [YOLO-World-Seg-M](./yolo_seg_world_m_dual_vlpan_2e-4_80e_8gpus_seghead_finetune_lvis.py) | `LVIS-Base` | `seg head` | 16.7 | 12.6 | 14.6 | 20.8  | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_seghead_finetune_lvis-7bca59a7.pth) |
| [YOLO-World-v2-Seg-M](./yolo_world_v2_seg_m_vlpan_bn_2e-4_80e_8gpus_seghead_finetune_lvis.py) | `LVIS-Base` | `seg head` | 17.8 | 13.9 | 15.5 | 22.0  | [HF Checkpoints 🤗]() |
| [YOLO-World-Seg-L](yolo_seg_world_l_dual_vlpan_2e-4_80e_8gpus_seghead_finetune_lvis.py) | `LVIS-Base` | `seg head` | 19.1 | 14.2 | 17.2 | 23.5 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_seg_l_dual_vlpan_2e-4_80e_8gpus_seghead_finetune_lvis-5a642d30.pth) |
| [YOLO-World-v2-Seg-L](./yolo_world_v2_seg_l_vlpan_bn_2e-4_80e_8gpus_seghead_finetune_lvis.py) | `LVIS-Base` | `seg head` | 19.8 | 17.2 | 17.5 | 23.6 | [HF Checkpoints 🤗]() |
**NOTE:**
1. The mask AP are evaluated on the LVIS `val 1.0`.
2. All models are fine-tuned for 80 epochs on `LVIS-Base` (866 categories, `common + frequent`).
3. The YOLO-World-Seg with only `seg head` fine-tuned maintains the original zero-shot detection capability and segments objects.
