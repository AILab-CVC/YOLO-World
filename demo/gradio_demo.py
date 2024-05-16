# Copyright (c) Tencent Inc. All rights reserved.
import os
import sys
import argparse
import os.path as osp
from io import BytesIO
from functools import partial

import cv2
import onnx
import torch
import onnxsim
import numpy as np
import gradio as gr
from PIL import Image
import supervision as sv
from torchvision.ops import nms
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmengine.config import Config, DictAction, ConfigDict
from mmdet.datasets import CocoDataset
from mmyolo.registry import RUNNERS

sys.path.append('./deploy')
from easydeploy import model as EM

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
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


def run_image(runner,
              image,
              text,
              max_num_boxes,
              score_thr,
              nms_thr,
              image_path='./work_dirs/demo.png'):
    # image.save(image_path)
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
    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None
    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    if masks is not None:
        image = MASK_ANNOTATOR.annotate(image, detections)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = Image.fromarray(image)
    return image


def export_model(runner, text, max_num_boxes, score_thr, nms_thr):

    backend = EM.MMYOLOBackend.ONNXRUNTIME
    postprocess_cfg = ConfigDict(pre_top_k=10 * max_num_boxes,
                                 keep_top_k=max_num_boxes,
                                 iou_threshold=nms_thr,
                                 score_threshold=score_thr)

    base_model = runner.model

    texts = [[t.strip() for t in text.split(',')] + [' ']]
    base_model.reparameterize(texts)
    deploy_model = EM.DeployModel(baseModel=base_model,
                                  backend=backend,
                                  postprocess_cfg=postprocess_cfg)
    deploy_model.eval()

    device = (next(iter(base_model.parameters()))).device
    fake_input = torch.ones([1, 3, 640, 640], device=device)
    deploy_model(fake_input)

    save_onnx_path = os.path.join(
        args.work_dir,
        os.path.basename(args.checkpoint).replace('pth', 'onnx'))
    # export onnx
    with BytesIO() as f:
        output_names = ['num_dets', 'boxes', 'scores', 'labels']
        torch.onnx.export(deploy_model,
                          fake_input,
                          f,
                          input_names=['images'],
                          output_names=output_names,
                          opset_version=12)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, save_onnx_path)
    return gr.update(visible=True), save_onnx_path


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
                with gr.Row():
                    export = gr.Button('Deploy and Export ONNX Model')
                with gr.Row():
                    gr.Markdown(
                        "It takes a few seconds to generate the ONNX file! YOLO-World-Seg (segmentation) is not supported now"
                    )
                out_download = gr.File(visible=False)
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
                output_image = gr.Image(type='pil', label='output image')

        submit.click(partial(run_image, runner),
                     [image, input_text, max_num_boxes, score_thr, nms_thr],
                     [output_image])
        clear.click(lambda: [None, '', None], None,
                    [image, input_text, output_image])

        export.click(partial(export_model, runner),
                     [input_text, max_num_boxes, score_thr, nms_thr],
                     [out_download, out_download])

        demo.launch(server_name='0.0.0.0',
                    server_port=8080)  # port 80 does not work for me


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
