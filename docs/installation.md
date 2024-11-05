## Installation Guide

YOLO-World is built based on `pytorch=1.11.0` and `mmcv=2.0.0`.

We provide the `requirements` files in [./requirements](./../requirements/):

* `basic_requirements`: training, finetuning, evaluation.
* `demo_requirements`: running YOLO-World [demos](./../demo/).
* `onnx_requirements`: converting YOLO-World to ONNX or TFLite models (TFLite is coming soon).

#### Install `MMCV`

YOLO-World adopts `mmcv>=2.0.0`. There are several ways to install `mmcv`

**1. using `openmim`**:

see more in [official guide](https://github.com/open-mmlab/mmcv/tree/master?tab=readme-ov-file#install-mmcv-full).

```bash
pip install openmim
mim install mmcv==2.0.0 
```

**2. using `pip`**:

go to [install-with-pip](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip) to select the pip index. 

```bash
# cuda=11.3, torch=1.11
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
# cuda=11.7, torch=1.13
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
# cuda=12.1, torch=2.1
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

**3. using `whl`**

go to [index packages](https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html) to find a suitable version and download.

```bash
pip install mmcv-2.0.1-cp38-cp38-manylinux1_x86_64.whl
```