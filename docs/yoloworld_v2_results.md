## Model Zoo

We've pre-trained YOLO-World-S/M/L from scratch and evaluate on the `LVIS val-1.0` and `LVIS minival`. We provide the pre-trained model weights and training logs for applications/research or re-producing the results.

### Zero-shot Inference on LVIS dataset

<div><font size=2>

| model                                                                                                                | Pre-train Data       | Size | AP<sup>mini</su> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | AP<sup>val</su> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> |                                                                                        weights                                                                                         |
| :------------------------------------------------------------------------------------------------------------------- | :------------------- | :----------------- | :--------------: | :------------: | :------------: | :------------: | :-------------: | :------------: | :------------: | :------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [YOLO-Worldv2-S](./configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG | 640 | 22.7 | 16.3 | 20.8 | 25.5 |  17.3 | 11.3 | 14.9 | 22.7 |[HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth)|
| [YOLO-Worldv2-S](./configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py) | O365+GoldG | 1280&#x1F538; | 24.1 | 18.7 | 22.0 | 26.9 |  18.8 | 14.1 | 16.3 | 23.8 |[HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_obj365v1_goldg_pretrain_1280ft-fc4ff4f7.pth)|  
| [YOLO-Worldv2-M](./configs/pretrain/yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py)  | O365+GoldG | 640 | 30.0 | 25.0  | 27.2 | 33.4 | 23.5 | 17.1 | 20.0 | 30.1 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth)| 
| [YOLO-Worldv2-M](./configs/pretrain/yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py) | O365+GoldG | 1280&#x1F538; | 31.6 | 24.5  | 29.0 | 35.1 | 25.3 | 19.3 | 22.0 | 31.7 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft-77d0346d.pth)| 
| [YOLO-Worldv2-L](./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG | 640 | 33.0 | 22.6 | 32.0 | 35.8 | 26.0 | 18.6 | 23.0 | 32.6 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth)| 
| 🔥 [YOLO-Worldv2-L](./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG+CCLiteV2 | 640 | 33.4 | 23.1 | 31.9 | 36.6 | 26.6 | 20.3 | 23.2 | 33.2 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_l_obj365v1_goldg_cc3mv2_pretrain-2f3a4a22.pth)| 
| [YOLO-Worldv2-L](./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py) | O365+GoldG | 1280&#x1F538; | 34.6 | 29.2 | 32.8 | 37.2 | 27.6 | 21.9 | 24.2 | 34.0 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth)| 
| [YOLO-Worldv2-L (CLIP-Large)](./configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) 🔥  | O365+GoldG | 640 | 34.0 | 22.0 | 32.6 | 37.4 | 27.1 | 19.9 | 23.9 | 33.9 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain-8ff2e744.pth)|
| [YOLO-Worldv2-L (CLIP-Large)](./configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_800ft_lvis_minival.py) 🔥  | O365+GoldG | 800&#x1F538; | 35.5 | 28.3 | 33.2 | 38.8 | 28.6 | 22.0 | 25.1 | 35.4 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth)|
| [YOLO-Worldv2-L](./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG+CC3M-Lite | 640 | 32.9 | 25.3 | 31.1 | 35.8 | 26.1 | 20.6 | 22.6 | 32.3 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth)|
| [YOLO-Worldv2-X](./configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG+CC3M-Lite | 640 | 35.4 | 28.7 | 32.9 | 38.7 | 28.4 | 20.6 | 25.6 | 35.0 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain-8698fbfa.pth) |
| 🔥 [YOLO-Worldv2-X]() |  O365+GoldG+CC3M-Lite | 1280&#x1F538; | 37.4 | 30.5 | 35.2 | 40.7  | 29.8 | 21.1 | 26.8 | 37.0 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth) |
| [YOLO-Worldv2-XL](./configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG+CC3M-Lite | 640 | 36.0 | 25.8 | 34.1 | 39.5 | 29.1 | 21.1 | 26.3 | 35.8 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth) |

</font>
</div>

**NOTE:**
1. AP<sup>mini</sup>: evaluated on LVIS `minival`.
3. AP<sup>val</sup>: evaluated on LVIS `val 1.0`.
4. [HuggingFace Mirror](https://hf-mirror.com/) provides the mirror of HuggingFace, which is a choice for users who are unable to reach.
5. &#x1F538;: fine-tuning models with the pre-trained data.

**Pre-training Logs:**

We provide the pre-training logs of `YOLO-World-v2`. Due to the unexpected errors of the local machines, the training might be interrupted several times.

| Model | YOLO-World-v2-S | YOLO-World-v2-M  | YOLO-World-v2-L | YOLO-World-v2-X |
| :---  | :-------------: | :--------------: | :-------------: | :-------------: |
|Pre-training Log | [Part-1](https://drive.google.com/file/d/1oib7pKfA2h1U_5-85H_s0Nz8jWd0R-WP/view?usp=drive_link), [Part-2](https://drive.google.com/file/d/11cZ6OZy80VTvBlZy3kzLAHCxx5Iix5-n/view?usp=drive_link) | [Part-1](https://drive.google.com/file/d/1E6vYSS8kBipGc8oQnsjAfeUAx8I9yOX7/view?usp=drive_link), [Part-2](https://drive.google.com/file/d/1fbM7vt2tgSeB8o_7tUDofWvpPNSViNj5/view?usp=drive_link) | [Part-1](https://drive.google.com/file/d/1Tola1QGJZTL6nGy3SBxKuknfNfREDm8J/view?usp=drive_link), [Part-2](https://drive.google.com/file/d/1mTBXniioUb0CdctCG4ckIU6idGo0NnH8/view?usp=drive_link) |  [Final part](https://drive.google.com/file/d/1aEUA_EPQbXOrpxHTQYB6ieGXudb1PLpd/view?usp=drive_link)| 

