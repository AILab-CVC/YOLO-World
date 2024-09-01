FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS DEPENDENCIES

ARG MODEL="yolo_world_l_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
ARG WEIGHT="yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth"

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip     \
    libgl1-mesa-glx \
    libsm6          \
    libxext6        \
    libxrender-dev  \
    libglib2.0-0    \
    git             \
    python3-dev     \
    python3-wheel   \
    curl

# Uncomment the following if you want to download a specific set of weights
# RUN mkdir weights
# RUN curl -o weights/$WEIGHT -L https://huggingface.co/wondervictor/YOLO-World/resolve/main/$WEIGHT

RUN pip3 install --upgrade pip \
    && pip3 install wheel \
    && pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install   \
        gradio==4.16.0 \
        opencv-python==4.9.0.80 \
        supervision \
        mmengine==0.10.4 \
        setuptools \
        openmim \
    && mim install mmcv==2.1.0 \
    && mim install mmdet==3.3.0 \
    && pip install git+https://github.com/onuralpszr/mmyolo.git

FROM DEPENDENCIES as INSTALLING_YOLO
RUN git clone --recursive https://github.com/tim-win/YOLO-World /yolo/
#COPY . /yolo
WORKDIR /yolo

RUN pip3 install -e .[demo]

RUN pip3 install onnx onnxsim

FROM INSTALLING_YOLO as OK_THIS_PART_IS_TRICKY_DONT_HATE

RUN mkdir /weights/
RUN chmod a+rwx /yolo/configs/*/*

CMD [ "bash" ]
