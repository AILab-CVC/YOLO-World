<div align="center">
<img src="./assets/logo.png" width=60%>
<br>
<a href="https://scholar.google.com/citations?hl=zh-CN&user=PH8rJHYAAAAJ">Tianheng Cheng*</a><sup><span>2,3</span></sup>, 
<a href="https://linsong.info/">Lin Song*</a><sup><span>1</span></sup>,
<a href="https://yxgeee.github.io/">Yixiao Ge</a><sup><span>1,2</span></sup>,
<a href="https://xwcv.github.io/">Xinggang Wang</a><sup><span>3</span></sup>,
<a href="http://eic.hust.edu.cn/professor/liuwenyu/"> Wenyu Liu</a><sup><span>3</span></sup>,
<a href="https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en">Ying Shan</a><sup><span>1,2</span></sup>
</br>

<sup>1</sup> Tencent AI Lab,  <sup>2</sup> ARC Lab, Tencent PCG
<sup>3</sup> Huazhong University of Science and Technology
<br>
<div>

[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)]([https://arxiv.org/abs/](https://arxiv.org/abs/2401.17270))
[![video](https://img.shields.io/badge/🤗HugginngFace-Spaces-orange)](https://huggingface.co/)
[![license](https://img.shields.io/badge/License-GPLv3.0-blue)](LICENSE)

</div>
</div>


## Updates 

`[2024-1-31]:` We are excited to launch **YOLO-World**, a cutting-edge real-time open-vocabulary object detector.

## Highlights

This repo contains the PyTorch implementation, pre-trained weights, and pre-training/fine-tuning code for YOLO-World.

* YOLO-World is pre-trained on large-scale datasets, including detection, grounding, and image-text datasets.

* YOLO-World is the next-generation YOLO detector, with a strong open-vocabulary detection capability and grounding ability.

* YOLO-World presents a *prompt-then-detect* paradigm for efficient user-vocabulary inference, which re-parameterizes vocabulary embeddings as parameters into the model and achieve superior inference speed. You can try to export your own detection model without extra training or fine-tuning in our [online demo]()!


<center>
<img width=800px src="./assets/yolo_arch.png">
</center>


## Abstract

The You Only Look Once (YOLO) series of detectors have established themselves as efficient and practical tools. However, their reliance on predefined and trained object categories limits their applicability in open scenarios. Addressing this limitation, we introduce YOLO-World, an innovative approach that enhances YOLO with open-vocabulary detection capabilities through vision-language modeling and pre-training on large-scale datasets. Specifically, we propose a new Re-parameterizable Vision-Language Path Aggregation Network (RepVL-PAN) and region-text contrastive loss to facilitate the interaction between visual and linguistic information. Our method excels in detecting a wide range of objects in a zero-shot manner with high efficiency. On the challenging LVIS dataset, YOLO-World achieves 35.4 AP with 52.0 FPS on V100, which outperforms many state-of-the-art methods in terms of both accuracy and speed. Furthermore, the fine-tuned YOLO-World achieves remarkable performance on several downstream tasks, including object detection and open-vocabulary instance segmentation.


## Demo


## Main Results

We've pre-trained YOLO-World-S/M/L from scratch and evaluate on the `LVIS val-1.0` and `LVIS minival`. We provide the pre-trained model weights and training logs for applications/research or re-producing the results.

### Zero-shot Inference on LVIS dataset

| model | Pre-train Data | AP | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | weights | log |
| :---- | :------------- | :-:| :------------: |:-------------: | :-------: | :---: | :---: |
| [YOLO-World-S]() | O365+GoldG | 26.2 | 19.1 | 23.6 | 29.8  | - | [coming soon] | [log]() |
| [YOLO-World-M]() | O365+GoldG | 31.0 | 23.8 | 29.2 | 33.9  | - | [coming soon] | [log]() |
| [YOLO-World-L]() | O365+GoldG | 35.0 | 27.1 | 32.8 | 38.3 | - | [coming soon]| [log]() |

**NOTE:**
1. The evaluation results are tested on LVIS minival in a zero-shot manner.


### Finetuning on COCO dataset


## Getting started

### 1. Installation

YOLO-World is developed based on `torch==1.11.0` `mmyolo==0.6.0` and `mmdetection==3.0.0`.

```bash
# install key dependencies
pip install mmdetection==3.0.0 mmengine transformers

# clone the repo
git clone https://xxxx.YOLO-World.git
cd YOLO-World 

# install mmyolo
mkdir third_party
git clone https://github.com/open-mmlab/mmyolo.git
cd ..

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

Evalutating YOLO-World is also easy:

```bash
chmod +x tools/dist_test.sh
./tools/dist_test.sh path/to/config path/to/weights 8
```

**NOTE:** We mainly evaluate the performance on LVIS-minival for pre-training.

## Deployment

We provide the details about deployment for downstream applications in [docs/deployment](./docs/deploy.md).
You can directly download the ONNX model through the online [demo]() in Huggingface Spaces 🤗.

## Acknowledgement

We sincerely thank [mmyolo](https://github.com/open-mmlab/mmyolo), [mmdetection](https://github.com/open-mmlab/mmdetection), and [transformers](https://github.com/huggingface/transformers) for providing their wonderful code to the community!

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
