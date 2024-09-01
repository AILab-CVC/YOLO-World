# Copyright (c) Tencent Inc. All rights reserved.
import os
import sys
import argparse
import os.path as osp
from functools import partial

import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
import supervision as sv
from torchvision.ops import nms
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmengine.config import Config, DictAction
from mmdet.datasets import CocoDataset
from mmyolo.registry import RUNNERS

sys.path.append('./deploy')

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Segmentation Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics',
        default='output')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def run_segmentation(runner,
                     image,
                     text,
                     max_num_boxes,
                     score_thr,
                     nms_thr):
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    data_info = dict(img_id=0, img=np.array(image), texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    keep = nms(pred_instances.bboxes,
               pred_instances.scores,
               iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    masks = pred_instances['masks'] if 'masks' in pred_instances else None
    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    if masks is not None:
        image = MASK_ANNOTATOR.annotate(image, detections)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image

def demo(runner, args):
    with gr.Blocks(title="YOLO-World Segmentation") as demo:
        gr.Markdown('<h1><center>YOLO-World: Real-Time Open-Vocabulary '
                    'Object Detector and Segmentation</center></h1>')
        with gr.Row():
            with gr.Column(scale=0.5):
                image = gr.Image(type='pil', label='Input Image')
                input_text = gr.Textbox(
                    lines=3,
                    label='Enter the classes to be detected and segmented, '
                    'separated by comma',
                    value=', '.join(CocoDataset.METAINFO['classes'][:5]),
                    elem_id='textbox')
                max_num_boxes = gr.Slider(minimum=1,
                                          maximum=300,
                                          value=100,
                                          step=1,
                                          interactive=True,
                                          label='Maximum Number of Boxes')
                score_thr = gr.Slider(minimum=0,
                                      maximum=1,
                                      value=0.3,
                                      step=0.01,
                                      interactive=True,
                                      label='Score Threshold')
                nms_thr = gr.Slider(minimum=0,
                                    maximum=1,
                                    value=0.5,
                                    step=0.01,
                                    interactive=True,
                                    label='NMS Threshold')
                submit = gr.Button('Submit')
                clear = gr.Button('Clear')
            with gr.Column(scale=0.5):
                output_image = gr.Image(type='pil', label='Output Image')

        submit.click(partial(run_segmentation, runner),
                     [image, input_text, max_num_boxes, score_thr, nms_thr],
                     [output_image])
        clear.click(lambda: [None, '', None], None,
                    [image, input_text, output_image])

        demo.launch(server_name='0.0.0.0', server_port=8080)

if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    runner.pipeline = Compose(pipeline)
    runner.model.eval()
    demo(runner, args)