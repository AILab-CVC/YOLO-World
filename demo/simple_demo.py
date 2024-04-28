# Copyright (c) Tencent Inc. All rights reserved.
import os.path as osp

import cv2
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


def inference(model, image, texts, test_pipeline, score_thr=0.3, max_dets=100):
    image = cv2.imread(image)
    image = image[:, :, [2, 1, 0]]
    data_info = dict(img=image, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    pred_instances = output.pred_instances
    # score thresholding
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    # max detections
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    boxes = pred_instances['bboxes']
    labels = pred_instances['labels']
    scores = pred_instances['scores']
    label_texts = [texts[x][0] for x in labels]
    return boxes, labels, label_texts, scores


if __name__ == "__main__":

    config_file = "configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
    checkpoint = "weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"

    cfg = Config.fromfile(config_file)
    cfg.work_dir = osp.join('./work_dirs')
    # init model
    cfg.load_from = checkpoint
    model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0')
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    texts = [['person'], ['bus'], [' ']]
    image = "demo/sample_images/bus.jpg"
    print(f"starting to detect: {image}")
    results = inference(model, image, texts, test_pipeline)
    format_str = [
        f"obj-{idx}: {box}, label-{lbl}, class-{lbl_text}, score-{score}"
        for idx, (box, lbl, lbl_text, score) in enumerate(zip(*results))
    ]
    print("detecting results:")
    for q in format_str:
        print(q)
