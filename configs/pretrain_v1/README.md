## Pre-training YOLO-World-v1

> The YOLO-World-v1 is an initial version, and now is nearly deprecated! We strongly suggest you use the [latest version](../pretrain/).



### Zero-shot Inference on LVIS dataset

| model                                                                                                                | Pre-train Data       | Size | AP<sup>mini</su> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | AP<sup>val</su> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> |                                                                                        weights                                                                                         |
| :------------------------------------------------------------------------------------------------------------------- | :------------------- | :----------------- | :--------------: | :------------: | :------------: | :------------: | :-------------: | :------------: | :------------: | :------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [YOLO-World-S](./yolo_world_s_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py)   | O365+GoldG           |        640        |       24.3       |      16.6      |      22.1      |      27.7      |      17.8       |      11.0      |      14.8      |      24.0      |    [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-18bea4d2.pth)    |
| [YOLO-World-M](./yolo_world_m_dual_l2norm_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py)   | O365+GoldG           |        640        |       28.6       |      19.7      |      26.6      |      31.9      |      22.3       |      16.2      |      19.0      |      28.7      |    [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_m_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-2b7bd1be.pth)    |
| [YOLO-World-L](./yolo_world_l_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py)   | O365+GoldG           |        640        |       32.5       |      22.3      |      30.6      |      36.1      |      24.8       |      17.8      |      22.4      |      32.5      |    [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth)    |
| [YOLO-World-L](./yolo_world_l_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG+CC3M-Lite |        640        |       33.0       |      23.6      |      32.0      |      35.5      |      25.3       |      18.0      |      22.1      |      32.1      | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-7a5eea3b.pth) |
| [YOLO-World-X](./yolo_world_x_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG+CC3M-Lite | 640 | 33.4 | 24.4 | 31.6 | 36.6 | 26.6 | 19.2 | 23.5 | 33.2 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_x_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-8cf6b025.pth) |


**NOTE:**
1. AP<sup>mini</sup>: evaluated on LVIS `minival`.
3. AP<sup>val</sup>: evaluated on LVIS `val 1.0`.
4. [HuggingFace Mirror](https://hf-mirror.com/) provides the mirror of HuggingFace, which is a choice for users who are unable to reach.