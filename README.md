<div align="center">
<img src="./assets/yolo_logo.png" width=60%>
<br>
<a href="https://scholar.google.com/citations?hl=zh-CN&user=PH8rJHYAAAAJ">Tianheng Cheng</a><sup><span>2,3,*</span></sup>, 
<a href="https://linsong.info/">Lin Song</a><sup><span>1,📧,*</span></sup>,
<a href="https://yxgeee.github.io/">Yixiao Ge</a><sup><span>1,🌟,2</span></sup>,
<a href="http://eic.hust.edu.cn/professor/liuwenyu/"> Wenyu Liu</a><sup><span>3</span></sup>,
<a href="https://xwcv.github.io/">Xinggang Wang</a><sup><span>3,📧</span></sup>,
<a href="https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en">Ying Shan</a><sup><span>1,2</span></sup>
</br>

\* Equal contribution 🌟 Project lead 📧 Corresponding author

<sup>1</sup> Tencent AI Lab,  <sup>2</sup> ARC Lab, Tencent PCG
<sup>3</sup> Huazhong University of Science and Technology
<br>
<div>

[![arxiv paper](https://img.shields.io/badge/Project-Page-green)](https://wondervictor.github.io/)
[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2401.17270)
<a href="https://colab.research.google.com/github/AILab-CVC/YOLO-World/blob/master/inference.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![demo](https://img.shields.io/badge/🤗HugginngFace-Spaces-orange)](https://huggingface.co/spaces/stevengrove/YOLO-World)
[![Replicate](https://replicate.com/zsxkib/yolo-world/badge)](https://replicate.com/zsxkib/yolo-world)
[![hfpaper](https://img.shields.io/badge/🤗HugginngFace-Paper-yellow)](https://huggingface.co/papers/2401.17270)
[![license](https://img.shields.io/badge/License-GPLv3.0-blue)](LICENSE)
[![yoloworldseg](https://img.shields.io/badge/YOLOWorldxEfficientSAM-🤗Spaces-orange)](https://huggingface.co/spaces/SkalskiP/YOLO-World)
[![yologuide](https://img.shields.io/badge/📖Notebook-roboflow-purple)](https://supervision.roboflow.com/develop/notebooks/zero-shot-object-detection-with-yolo-world)
[![deploy](https://media.roboflow.com/deploy.svg)](https://inference.roboflow.com/foundation/yolo_world/)

</div>
</div>

## Notice

We recommend that everyone **use English to communicate on issues**, as this helps developers from around the world discuss, share experiences, and answer questions together.

## 🔥 Updates 
`[2024-3-16]:` We fix the bugs about the demo ([#110](https://github.com/AILab-CVC/YOLO-World/issues/110),[#94](https://github.com/AILab-CVC/YOLO-World/issues/94),[#129](https://github.com/AILab-CVC/YOLO-World/issues/129), [#125](https://github.com/AILab-CVC/YOLO-World/issues/125)) with visualizations of segmentation masks, and release [**YOLO-World with Embeddings**](./docs/prompt_yolo_world.md), which supports prompt tuning, text prompts and image prompts.  
`[2024-3-3]:` We add the **high-resolution YOLO-World**, which supports `1280x1280` resolution with higher accuracy and better performance for small objects!  
`[2024-2-29]:` We release the newest version of [ **YOLO-World-v2**](./docs/updates.md) with higher accuracy and faster speed! We hope the community can join us to improve YOLO-World!  
`[2024-2-28]:` Excited to announce that YOLO-World has been accepted by **CVPR 2024**! We're continuing to make YOLO-World faster and stronger, as well as making it better to use for all.  
`[2024-2-22]:` We sincerely thank [RoboFlow](https://roboflow.com/) and [@Skalskip92](https://twitter.com/skalskip92) for the [**Video Guide**](https://www.youtube.com/watch?v=X7gKBGVz4vs) about YOLO-World, nice work!  
`[2024-2-18]:` We thank [@Skalskip92](https://twitter.com/skalskip92) for developing the wonderful segmentation demo via connecting YOLO-World and EfficientSAM. You can try it now at the [🤗 HuggingFace Spaces](https://huggingface.co/spaces/SkalskiP/YOLO-World).   
`[2024-2-17]:` The largest model **X** of YOLO-World is released, which achieves better zero-shot performance!   
`[2024-2-17]:` We release the code & models for **YOLO-World-Seg** now! YOLO-World now supports open-vocabulary / zero-shot object segmentation!  
`[2024-2-15]:` The pre-traind YOLO-World-L with CC3M-Lite is released!     
`[2024-2-14]:` We provide the [`image_demo`](demo.py) for inference on images or directories.   
`[2024-2-10]:` We provide the [fine-tuning](./docs/finetuning.md) and [data](./docs/data.md) details for fine-tuning YOLO-World on the COCO dataset or the custom datasets!  
`[2024-2-3]:` We support the `Gradio` demo now in the repo and you can build the YOLO-World demo on your own device!  
`[2024-2-1]:` We've released the code and weights of YOLO-World now!  
`[2024-2-1]:` We deploy the YOLO-World demo on [HuggingFace 🤗](https://huggingface.co/spaces/stevengrove/YOLO-World), you can try it now!  
`[2024-1-31]:` We are excited to launch **YOLO-World**, a cutting-edge real-time open-vocabulary object detector.  


## TODO

YOLO-World is under active development and please stay tuned ☕️! 
If you have suggestions📃 or ideas💡,**we would love for you to bring them up in the [Roadmap](https://github.com/AILab-CVC/YOLO-World/issues/109)** ❤️!
> YOLO-World 目前正在积极开发中📃，如果你有建议或者想法💡，**我们非常希望您在 [Roadmap](https://github.com/AILab-CVC/YOLO-World/issues/109) 中提出来** ❤️！

## [FAQ (Frequently Asked Questions)](https://github.com/AILab-CVC/YOLO-World/discussions/149)

We have set up an FAQ about YOLO-World in the discussion on GitHub. We hope everyone can raise issues or solutions during use here, and we also hope that everyone can quickly find solutions from it.

> 我们在GitHub的discussion中建立了关于YOLO-World的常见问答，这里将收集一些常见问题，同时大家可以在此提出使用中的问题或者解决方案，也希望大家能够从中快速寻找到解决方案


## Highlights & Introduction

This repo contains the PyTorch implementation, pre-trained weights, and pre-training/fine-tuning code for YOLO-World.

* YOLO-World is pre-trained on large-scale datasets, including detection, grounding, and image-text datasets.

* YOLO-World is the next-generation YOLO detector, with a strong open-vocabulary detection capability and grounding ability.

* YOLO-World presents a *prompt-then-detect* paradigm for efficient user-vocabulary inference, which re-parameterizes vocabulary embeddings as parameters into the model and achieve superior inference speed. You can try to export your own detection model without extra training or fine-tuning in our [online demo](https://huggingface.co/spaces/stevengrove/YOLO-World)!


<center>
<img width=800px src="./assets/yolo_arch.png">
</center>

## Model Zoo

We've pre-trained YOLO-World-S/M/L from scratch and evaluate on the `LVIS val-1.0` and `LVIS minival`. We provide the pre-trained model weights and training logs for applications/research or re-producing the results.

### Zero-shot Inference on LVIS dataset

| model                                                                                                                | Pre-train Data       | Size | AP<sup>mini</su> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | AP<sup>val</su> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> |                                                                                        weights                                                                                         |
| :------------------------------------------------------------------------------------------------------------------- | :------------------- | :----------------- | :--------------: | :------------: | :------------: | :------------: | :-------------: | :------------: | :------------: | :------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [YOLO-World-S](./configs/pretrain/yolo_world_s_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py)   | O365+GoldG           |        640        |       24.3       |      16.6      |      22.1      |      27.7      |      17.8       |      11.0      |      14.8      |      24.0      |    [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-18bea4d2.pth)    |
| [YOLO-Worldv2-S](./configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) 🔥  | O365+GoldG | 640 | 22.7 | 16.3 | 20.8 | 25.5 |  17.3 | 11.3 | 14.9 | 22.7 |[HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth)| 
| [YOLO-World-M](./configs/pretrain/yolo_world_m_dual_l2norm_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py)   | O365+GoldG           |        31.0        |       28.6       |      19.7      |      26.6      |      31.9      |      22.3       |      16.2      |      19.0      |      28.7      |    [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_m_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-2b7bd1be.pth)    |
| [YOLO-Worldv2-M](./configs/pretrain/yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) 🔥  | O365+GoldG | 640 | 30.0 | 25.0  | 27.2 | 33.4 | 23.5 | 17.1 | 20.0 | 30.1 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth)| 
| [YOLO-World-L](./configs/pretrain/yolo_world_l_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py)   | O365+GoldG           |        640        |       32.5       |      22.3      |      30.6      |      36.1      |      24.8       |      17.8      |      22.4      |      32.5      |    [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth)    |
| [YOLO-World-L](./configs/pretrain/yolo_world_l_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG+CC3M-Lite |        640        |       33.0       |      23.6      |      32.0      |      35.5      |      25.3       |      18.0      |      22.1      |      32.1      | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-7a5eea3b.pth) |
| [YOLO-Worldv2-L](./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) 🔥  | O365+GoldG | 640 | 33.0 | 22.6 | 32.0 | 35.8 | 26.0 | 18.6 | 23.0 | 32.6 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth)| 
| [YOLO-Worldv2-L](./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py) 🔥  | O365+GoldG | 1280 &#x1F538; | 34.6 | 29.2 | 32.8 | 37.2 | 27.6 | 21.9 | 24.2 | 34.0 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth)| 
| [YOLO-Worldv2-L](./configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) 🔥 | O365+GoldG+CC3M-Lite | 640 | 32.9 | 25.3 | 31.1 | 35.8 | 26.1 | 20.6 | 22.6 | 32.3 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth)|
| [YOLO-World-X](./configs/pretrain/yolo_world_x_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG+CC3M-Lite | 640 | 33.4 | 24.4 | 31.6 | 36.6 | 26.6 | 19.2 | 23.5 | 33.2 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_x_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-8cf6b025.pth) |
| [YOLO-Worldv2-X](./configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) 🔥  | O365+GoldG+CC3M-Lite | 640 | 35.4 | 28.7 | 32.9 | 38.7 | 28.4 | 20.6 | 25.6 | 35.0 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain-8698fbfa.pth) |

**NOTE:**
1. AP<sup>mini</sup>: evaluated on LVIS `minival`.
3. AP<sup>val</sup>: evaluated on LVIS `val 1.0`.
4. [HuggingFace Mirror](https://hf-mirror.com/) provides the mirror of HuggingFace, which is a choice for users who are unable to reach.
5. &#x1F538;: fine-tuning models with the pre-trained data.

**Pre-training Logs:**

We provide the pre-training logs of `YOLO-World-v2`. Due to the unexpected errors of the local machines, the training might be interrupted several times.

| Model | Pre-training Log |
| :---: | :--------------: |
| YOLO-World-v2-S | [Part-1](https://drive.google.com/file/d/1oib7pKfA2h1U_5-85H_s0Nz8jWd0R-WP/view?usp=drive_link), [Part-2](https://drive.google.com/file/d/11cZ6OZy80VTvBlZy3kzLAHCxx5Iix5-n/view?usp=drive_link) |
| YOLO-World-v2-L | [Part-1](https://drive.google.com/file/d/1Tola1QGJZTL6nGy3SBxKuknfNfREDm8J/view?usp=drive_link), [Part-2](https://drive.google.com/file/d/1mTBXniioUb0CdctCG4ckIU6idGo0NnH8/view?usp=drive_link) |
| YOLO-World-v2-M | [Part-1](https://drive.google.com/file/d/1E6vYSS8kBipGc8oQnsjAfeUAx8I9yOX7/view?usp=drive_link), [Part-2](https://drive.google.com/file/d/1fbM7vt2tgSeB8o_7tUDofWvpPNSViNj5/view?usp=drive_link) |
| YOLO-World-v2-X | [Final part](https://drive.google.com/file/d/1aEUA_EPQbXOrpxHTQYB6ieGXudb1PLpd/view?usp=drive_link) |

## Getting started

### 1. Installation

YOLO-World is developed based on `torch==1.11.0` `mmyolo==0.6.0` and `mmdetection==3.0.0`.

#### Clone Project 

```bash
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
```
#### Install

```bash
pip install torch wheel -q
pip install -e .
```

### 2. Preparing Data

We provide the details about the pre-training data in [docs/data](./docs/data.md).


## Training & Evaluation

We adopt the default [training](./tools/train.py) or [evaluation](./tools/test.py) scripts of [mmyolo](https://github.com/open-mmlab/mmyolo).
We provide the configs for pre-training and fine-tuning in `configs/pretrain` and `configs/finetune_coco`.
Training YOLO-World is easy:

```bash
chmod +x tools/dist_train.sh
# sample command for pre-training, use AMP for mixed-precision training
./tools/dist_train.sh configs/pretrain/yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py 8 --amp
```
**NOTE:** YOLO-World is pre-trained on 4 nodes with 8 GPUs per node (32 GPUs in total). For pre-training, the `node_rank` and `nnodes` for multi-node training should be specified. 

Evaluating YOLO-World is also easy:

```bash
chmod +x tools/dist_test.sh
./tools/dist_test.sh path/to/config path/to/weights 8
```

**NOTE:** We mainly evaluate the performance on LVIS-minival for pre-training.

## Fine-tuning YOLO-World

We provide the details about fine-tuning YOLO-World in [docs/fine-tuning](./docs/finetuning.md).

## Deployment

We provide the details about deployment for downstream applications in [docs/deployment](./docs/deploy.md).
You can directly download the ONNX model through the online [demo](https://huggingface.co/spaces/stevengrove/YOLO-World) in Huggingface Spaces 🤗.

## Demo

### Gradio Demo

We provide the [Gradio](https://www.gradio.app/) demo for local devices:

```bash
pip install gradio==4.16.0
python demo.py path/to/config path/to/weights
```

Additionaly, you can use a Dockerfile to build an image with gradio. As a prerequisite, make sure you have respective drivers installed alongside [nvidia-container-runtime](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime). Replace MODEL_NAME and WEIGHT_NAME with the respective values or ommit this and use default values from the [Dockerfile](Dockerfile#3)

```bash
docker build --build-arg="MODEL=MODEL_NAME" --build-arg="WEIGHT=WEIGHT_NAME" -t yolo_demo .
docker run --runtime nvidia -p 8080:8080
```

### Image Demo

We provide a simple image demo for inference on images with visualization outputs.

```bash
python image_demo.py path/to/config path/to/weights image/path/directory 'person,dog,cat' --topk 100 --threshold 0.005 --output-dir demo_outputs
```

**Notes:**
* The `image` can be a directory or a single image.
* The `texts` can be a string of categories (noun phrases) which is separated by a comma. We also support `txt` file in which each line contains a category ( noun phrases).
* The `topk` and `threshold` control the number of predictions and the confidence threshold.

### Google Golab Notebook

We sincerely thank [Onuralp](https://github.com/onuralpszr) for sharing the [Colab Demo](https://colab.research.google.com/drive/1F_7S5lSaFM06irBCZqjhbN7MpUXo6WwO?usp=sharing), you can have a try 😊！


## Acknowledgement

We sincerely thank [mmyolo](https://github.com/open-mmlab/mmyolo), [mmdetection](https://github.com/open-mmlab/mmdetection), [GLIP](https://github.com/microsoft/GLIP), and [transformers](https://github.com/huggingface/transformers) for providing their wonderful code to the community!

## Citations
If you find YOLO-World is useful in your research or applications, please consider giving us a star 🌟 and citing it.

```bibtex
@article{cheng2024yolow,
  title={YOLO-World: Real-Time Open-Vocabulary Object Detection},
  author={Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying},
  journal={arXiv preprint arXiv:2401.17270},
  year={2024}
}
```

## Licence
YOLO-World is under the GPL-v3 Licence and is supported for comercial usage.
