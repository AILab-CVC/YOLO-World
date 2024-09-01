# Base image with CUDA support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

# Set environment variables
ENV FORCE_CUDA="1" \
    MMCV_WITH_OPS=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    git \
    python3-dev \
    python3-wheel \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
FROM base AS python_deps

RUN pip3 install --upgrade pip wheel \
    && pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
        gradio==4.16.0 \
        opencv-python==4.9.0.80 \
        supervision \
        mmengine==0.10.4 \
        setuptools \
        openmim \
        onnx \
        onnxsim \
    && mim install mmcv==2.1.0 \
    && mim install mmdet==3.3.0 \
    && pip3 install --no-cache-dir git+https://github.com/onuralpszr/mmyolo.git

# Clone and install YOLO-World
FROM python_deps AS yolo_world

RUN git clone --recursive https://github.com/AILab-CVC/YOLO-World /yolo/
WORKDIR /yolo

RUN pip3 install -e .[demo]

# Final stage
FROM yolo_world AS final

ARG MODEL="yolo_world_l_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
ARG WEIGHT="yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth"

# Create weights directory and set permissions
RUN mkdir /weights/ \
    && chmod a+rwx /yolo/configs/*/*

# Optionally download weights (commented out by default)
# RUN curl -o /weights/$WEIGHT -L https://huggingface.co/wondervictor/YOLO-World/resolve/main/$WEIGHT

# Set the default command
CMD ["bash"]