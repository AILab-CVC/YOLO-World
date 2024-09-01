#!/usr/bin/env bash
set -e

export MODEL=yolo_world_v2_seg_l_vlpan_bn_2e-4_80e_8gpus_seghead_finetune_lvis.py
export WEIGHT=yolo_world_seg_l_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis-8c58c916.pth

export MODEL=yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_cc3mlite_train_lvis_minival.py
export WEIGHT=yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain-8698fbfa.pth

docker build -f ./Dockerfile --build-arg="MODEL=$MODEL" --build-arg="WEIGHT=$WEIGHT" -t yolo-demo . && docker run --runtime nvidia -p 8080:8080 yolo-demo python3 demo/gradio_demo.py "configs/pretrain/$MODEL" "/weights/$WEIGHT"