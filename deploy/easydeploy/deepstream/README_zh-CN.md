# 使用 DeepStream SDK 推理 MMYOLO 模型

本项目演示了如何使用 [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) 配合改写的 parser 来推理 MMYOLO 的模型。

## 预先准备

### 1. 安装 Nidia 驱动和 CUDA

首先请根据当前的显卡驱动和目标使用设备的驱动完成显卡驱动和 CUDA 的安装。

### 2. 安装 DeepStream SDK

目前 DeepStream SDK 稳定版本已经更新到 v6.2，官方推荐使用这个版本。

### 3. 将 MMYOLO 模型转换为 TensorRT Engine

推荐使用 EasyDeploy 中的 TensorRT 方案完成目标模型的转换部署，具体可参考 [此文档](../../easydeploy/docs/model_convert.md) 。

## 编译使用

当前项目使用的是 MMYOLO 的 rtmdet 模型，若想使用其他的模型，请参照目录下的配置文件进行改写。然后将转换完的 TensorRT engine 放在当前目录下并执行如下命令：

```bash
mkdir build && cd build
cmake ..
make -j$(nproc) && make install
```

完成编译后可使用如下命令进行推理：

```bash
deepstream-app -c deepstream_app_config.txt
```

## 项目代码结构

```bash
├── deepstream
│   ├── configs                   # MMYOLO 模型对应的 DeepStream 配置
│   │   └── config_infer_rtmdet.txt
│   ├── custom_mmyolo_bbox_parser # 适配 DeepStream formats 的 parser
│   │   └── nvdsparsebbox_mmyolo.cpp
|   ├── CMakeLists.txt
│   ├── coco_labels.txt           # coco labels
│   ├── deepstream_app_config.txt # DeepStream app 配置
│   ├── README_zh-CN.md
│   └── README.md
```
