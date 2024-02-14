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
[![demo](https://img.shields.io/badge/🤗HugginngFace-Spaces-orange)](https://huggingface.co/spaces/stevengrove/YOLO-World)
[![hfpaper](https://img.shields.io/badge/🤗HugginngFace-Paper-yellow)](https://huggingface.co/papers/2401.17270)
[![license](https://img.shields.io/badge/License-GPLv3.0-blue)](LICENSE)

</div>
</div>


## Updates 
`🔥[2024-2-14]:` We provide the [`image_demo`](demo.py) for inference on images or directories.   
`🔥[2024-2-10]:` We provide the [fine-tuning](./docs/finetuning.md) and [data](./docs/data.md) details for fine-tuning YOLO-World on the COCO dataset or the custom datasets!  
`[2024-2-3]:` We support the `Gradio` demo now in the repo and you can build the YOLO-World demo on your own device!  
`[2024-2-1]:` We've released the code and weights of YOLO-World now!  
`[2024-2-1]:` We deploy the YOLO-World demo on [HuggingFace 🤗](https://huggingface.co/spaces/stevengrove/YOLO-World), you can try it now!  
`[2024-1-31]:` We are excited to launch **YOLO-World**, a cutting-edge real-time open-vocabulary object detector.  



## TODO

YOLO-World is under active development and please stay tuned ☕️!

- [x] Gradio demo!
- [x] Complete documents for pre-training YOLO-World.
- [x] COCO & LVIS fine-tuning.
- [ ] Extra pre-trained models on more data, such as CC3M.
- [ ] Deployment toolkits, e.g., ONNX or TensorRT. 
- [ ] Inference acceleration and scripts for speed evaluation.
- [ ] Automatic labeling framework for image-text pairs, such as CC3M.


## Highlights

This repo contains the PyTorch implementation, pre-trained weights, and pre-training/fine-tuning code for YOLO-World.

* YOLO-World is pre-trained on large-scale datasets, including detection, grounding, and image-text datasets.

* YOLO-World is the next-generation YOLO detector, with a strong open-vocabulary detection capability and grounding ability.

* YOLO-World presents a *prompt-then-detect* paradigm for efficient user-vocabulary inference, which re-parameterizes vocabulary embeddings as parameters into the model and achieve superior inference speed. You can try to export your own detection model without extra training or fine-tuning in our [online demo](https://huggingface.co/spaces/stevengrove/YOLO-World)!


<center>
<img width=800px src="./assets/yolo_arch.png">
</center>


## Abstract

The You Only Look Once (YOLO) series of detectors have established themselves as efficient and practical tools. However, their reliance on predefined and trained object categories limits their applicability in open scenarios. Addressing this limitation, we introduce YOLO-World, an innovative approach that enhances YOLO with open-vocabulary detection capabilities through vision-language modeling and pre-training on large-scale datasets. Specifically, we propose a new Re-parameterizable Vision-Language Path Aggregation Network (RepVL-PAN) and region-text contrastive loss to facilitate the interaction between visual and linguistic information. Our method excels in detecting a wide range of objects in a zero-shot manner with high efficiency. On the challenging LVIS dataset, YOLO-World achieves 35.4 AP with 52.0 FPS on V100, which outperforms many state-of-the-art methods in terms of both accuracy and speed. Furthermore, the fine-tuned YOLO-World achieves remarkable performance on several downstream tasks, including object detection and open-vocabulary instance segmentation.


## Main Results

We've pre-trained YOLO-World-S/M/L from scratch and evaluate on the `LVIS val-1.0` and `LVIS minival`. We provide the pre-trained model weights and training logs for applications/research or re-producing the results.

### Zero-shot Inference on LVIS dataset

| model | Pre-train Data | AP<sup>fixed</sup> | AP<sup>mini</su> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | AP<sup>val</su> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | weights |
| :---- | :------------- | :----------------: | :---------------:| :------------: |:-------------: | :------------: | :-:| :------------: |:-------------: | :------------:  | :---: |
| [YOLO-World-S](./configs/pretrain/yolo_world_s_dual_3block_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG | 26.2 | 24.3 | 16.6 | 22.1 | 27.7 | 17.8 | 11.0 | 14.8 | 24.0 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-18bea4d2.pth) |
| [YOLO-World-M](./configs/pretrain/yolo_world_m_dual_l2norm_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG | 31.0 | 28.6 | 19.7 | 26.6 | 31.9 | 22.3 | 16.2 | 19.0 | 28.7 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_m_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-2b7bd1be.pth) |
| [YOLO-World-L](./configs/pretrain/yolo_world_s_dual_3block_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | O365+GoldG | 35.0 | 32.5 | 22.3 | 30.6 | 36.1 | 24.8 | 17.8 | 22.4 | 32.5 | [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth)  | 

**NOTE:**
1. The evaluation results of AP<sup>fixed</sup> are tested on LVIS `minival` with [fixed AP](https://github.com/achalddave/large-vocab-devil).
2. The evaluation results of AP<sup>mini</sup> are tested on LVIS `minival`.
3. The evaluation results of AP<sup>val</sup> are tested on LVIS `val 1.0`.

## Getting started

### 1. Installation

YOLO-World is developed based on `torch==1.11.0` `mmyolo==0.6.0` and `mmdetection==3.0.0`.

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
pip install gradio
python demo.py path/to/config path/to/weights
```

### Image Demo

We provide a simple image demo for inference on images with visualization outputs.

```bash
python image_demo.py path/to/config path/to/weights image/path/directory 'person,dog,cat' --topk 100 --threshold 0.005 --output-dir demo_outputs
```

### Google Golab Notebook

We sincerely thank [Onuralp](https://github.com/onuralpszr) for sharing the [Colab Demo](https://colab.research.google.com/drive/1F_7S5lSaFM06irBCZqjhbN7MpUXo6WwO?usp=sharing), you can have a try 😊！

**Notes:**
* The `image` can be a directory or a single image.
* The `texts` can be a string of categories (noun phrases) which is separated by a comma. We also support `txt` file in which each line contains a category ( noun phrases).
* The `topk` and `threshold` control the number of predictions and the confidence threshold.

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
