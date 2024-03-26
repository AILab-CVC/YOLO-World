# Inference MMYOLO Models with DeepStream

This project demonstrates how to inference MMYOLO models with customized parsers in [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk).

## Pre-requisites

### 1. Install Nvidia Driver and CUDA

First, please follow the official documents and instructions to install dedicated Nvidia graphic driver and CUDA matched to your gpu and target Nvidia AIoT devices.

### 2. Install DeepStream SDK

Second, please follow the official instruction to download and install DeepStream SDK. Currently stable version of DeepStream is v6.2.

### 3. Generate TensorRT Engine

As DeepStream builds on top of several NVIDIA libraries, you need to first convert your trained MMYOLO models to TensorRT engine files. We strongly recommend you to try the supported TensorRT deployment solution in [EasyDeploy](../../easydeploy/).

## Build and Run

Please make sure that your converted TensorRT engine is already located in the `deepstream` folder as the config shows. Create your own model config files and change the `config-file` parameter in [deepstream_app_config.txt](deepstream_app_config.txt) to the model you want to run with.

```bash
mkdir build && cd build
cmake ..
make -j$(nproc) && make install
```

Then you can run the inference with this command.

```bash
deepstream-app -c deepstream_app_config.txt
```

## Code Structure

```bash
├── deepstream
│   ├── configs                   # config file for MMYOLO models
│   │   └── config_infer_rtmdet.txt
│   ├── custom_mmyolo_bbox_parser # customized parser for MMYOLO models to DeepStream formats
│   │   └── nvdsparsebbox_mmyolo.cpp
|   ├── CMakeLists.txt
│   ├── coco_labels.txt           # labels for coco detection
│   ├── deepstream_app_config.txt # deepStream reference app configs for MMYOLO models
│   ├── README_zh-CN.md
│   └── README.md
```
