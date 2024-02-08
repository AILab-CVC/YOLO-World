# Copyright (c) Tencent Inc. All rights reserved.
import argparse
import os.path as osp
from functools import partial

import cv2
import torch
import gradio as gr
import numpy as np
import supervision as sv
from PIL import Image
from torchvision.ops import nms
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmdet.visualization import DetLocalVisualizer
from mmdet.datasets import CocoDataset
from mmyolo.registry import RUNNERS


BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
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


def run_image(runner,
              image,
              text,
              max_num_boxes,
              score_thr,
              nms_thr,
              image_path='./work_dirs/demo.png'):
    image.save(image_path)
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert RGB to BGR
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    image = Image.fromarray(image)
    return image


def demo(runner, args):
    with gr.Blocks(title="YOLO-World") as demo:
        with gr.Row():
            gr.Markdown('<h1><center>YOLO-World: Real-Time Open-Vocabulary '
                        'Object Detector</center></h1>')
        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Row():
                    image = gr.Image(type='pil', label='input image')
                input_text = gr.Textbox(
                    lines=7,
                    label='Enter the classes to be detected, '
                    'separated by comma',
                    value=', '.join(CocoDataset.METAINFO['classes']),
                    elem_id='textbox')
                with gr.Row():
                    submit = gr.Button('Submit')
                    clear = gr.Button('Clear')
                max_num_boxes = gr.Slider(minimum=1,
                                          maximum=300,
                                          value=100,
                                          step=1,
                                          interactive=True,
                                          label='Maximum Number Boxes')
                score_thr = gr.Slider(minimum=0,
                                      maximum=1,
                                      value=0.05,
                                      step=0.001,
                                      interactive=True,
                                      label='Score Threshold')
                nms_thr = gr.Slider(minimum=0,
                                    maximum=1,
                                    value=0.7,
                                    step=0.001,
                                    interactive=True,
                                    label='NMS Threshold')
            with gr.Column(scale=0.7):
                output_image = gr.Image(type='pil',
                                        label='output image')

        submit.click(partial(run_image, runner),
                     [image, input_text, max_num_boxes, score_thr, nms_thr],
                     [output_image])
        clear.click(lambda: [[], '', ''], None,
                    [image, input_text, output_image])
        demo.launch(server_name='0.0.0.0', server_port=80)


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
    runner.pipeline = Compose(pipeline)
    runner.model.eval()
    demo(runner, args)
