#!/usr/bin/env bash
set -e

MODEL_DIR="../models/models-yoloworld"

declare -A models
models["seg-l"]="yolo_world_v2_seg_l_vlpan_bn_2e-4_80e_8gpus_seghead_finetune_lvis.py yolo_world_seg_l_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis-8c58c916.pth"
models["pretrain-l-clip-800ft"]="yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_800ft_lvis_minival.py yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth"
models["pretrain-l-clip"]="yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py yolo_world_v2_l_clip_large_o365v1_goldg_pretrain-8ff2e744.pth"
models["pretrain-l-1280ft"]="yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
models["pretrain-l"]="yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth"
models["pretrain-m-1280ft"]="yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft-77d0346d.pth"
models["pretrain-m"]="yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth"
models["pretrain-s-1280ft"]="yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py yolo_world_v2_s_obj365v1_goldg_pretrain_1280ft-fc4ff4f7.pth"
models["pretrain-s"]="yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth"
models["pretrain-x-cc3mlite"]="yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_cc3mlite_train_lvis_minival.py yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain-8698fbfa.pth"
models["pretrain-x-1280ft"]="yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"

if [ $# -eq 0 ]; then
    echo "Available model keys:"
    for key in "${!models[@]}"; do
        echo "  $key"
    done
    echo "Usage: $0 <model-key>"
    exit 1
fi

model_key=$1

if [ -z "${models[$model_key]}" ]; then
    echo "Invalid model key. Available keys are:"
    for key in "${!models[@]}"; do
        echo "  $key"
    done
    exit 1
fi

read MODEL WEIGHT <<< "${models[$model_key]}"

config_dir="configs/pretrain"
demo_file=demo/gradio_demo.py
if [[ $model_key == seg-* ]]; then
    config_dir="configs/segmentation"
    demo_file="demo/segmentation_demo.py"
fi

docker build -f ./Dockerfile --build-arg="MODEL=$MODEL" --build-arg="WEIGHT=$WEIGHT" -t "yolo-demo:$model_key" . && \
docker run -it -v "$MODEL_DIR:/weights/" --runtime nvidia -p 8080:8080 "yolo-demo:$model_key" bash  # python3 demo/gradio_demo.py "$config_dir/$MODEL" "/weights/$WEIGHT"