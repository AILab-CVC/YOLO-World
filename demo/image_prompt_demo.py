# Copyright (c) Tencent Inc. All rights reserved.
import os
import sys
import argparse
import os.path as osp
from io import BytesIO
from functools import partial

import cv2
# import onnx
import torch
# import onnxsim
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

from transformers import (AutoTokenizer, CLIPTextModelWithProjection)
from transformers import (AutoProcessor, CLIPVisionModelWithProjection)

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
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


def generate_image_embeddings(prompt_image,
                              vision_encoder,
                              vision_processor,
                              projector,
                              device='cuda:0'):
    prompt_image = prompt_image.convert('RGB')
    inputs = vision_processor(images=[prompt_image],
                              return_tensors="pt",
                              padding=True)
    inputs = inputs.to(device)
    image_outputs = vision_encoder(**inputs)
    img_feats = image_outputs.image_embeds.view(1, -1)
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    if projector is not None:
        img_feats = projector(img_feats)
    return img_feats


def run_image(runner,
              vision_encoder,
              vision_processor,
              padding_token,
              image,
              text,
              prompt_image,
              add_padding,
              max_num_boxes,
              score_thr,
              nms_thr,
              image_path='./work_dirs/demo.png'):
    image = image.convert('RGB')
    if prompt_image is not None:
        texts = [['object'], [' ']]
        projector = None
        if hasattr(runner.model, 'image_prompt_encoder'):
            projector = runner.model.image_prompt_encoder.projector
        prompt_embeddings = generate_image_embeddings(
            prompt_image,
            vision_encoder=vision_encoder,
            vision_processor=vision_processor,
            projector=projector)
        if add_padding == 'padding':
            prompt_embeddings = torch.cat([prompt_embeddings, padding_token],
                                          dim=0)
        prompt_embeddings = prompt_embeddings / prompt_embeddings.norm(
            p=2, dim=-1, keepdim=True)
        runner.model.num_test_classes = prompt_embeddings.shape[0]
        runner.model.setembeddings(prompt_embeddings[None])
    else:
        runner.model.setembeddings(None)
        texts = [[t.strip()] for t in text.split(',')]
    data_info = dict(img_id=0, img=np.array(image), texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=False), torch.no_grad():
        if (prompt_image is not None) and ('texts' in data_batch['data_samples'][
                0]):
            del data_batch['data_samples'][0]['texts']
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


def demo(runner, args, vision_encoder, vision_processor, padding_embed):
    with gr.Blocks(title="YOLO-World") as demo:
        with gr.Row():
            gr.Markdown('<h1><center>YOLO-World: Real-Time Open-Vocabulary '
                        'Object Detector</center></h1>')
        with gr.Row():
            image = gr.Image(type='pil', label='input image')
            output_image = gr.Image(type='pil', label='output image')
        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Row():
                    prompt_image = gr.Image(type='pil',
                                            label='Image Prompts',
                                            height=300)
                with gr.Row():
                    add_padding = gr.Radio(["padding", "none"],
                                           label="Padding Prompt",
                                           info="whether add padding prompt")
            with gr.Column(scale=0.3):
                with gr.Row():
                    input_text = gr.Textbox(
                        lines=7,
                        label='Text Prompts:\nEnter the classes to be detected, '
                        'separated by comma',
                        value=', '.join(CocoDataset.METAINFO['classes']),
                        elem_id='textbox')
            with gr.Column(scale=0.4):
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

                with gr.Row():
                    submit = gr.Button('Submit')
                    clear = gr.Button('Clear')

        exp_image_dir = "./gradio_examples/image_prompts/images/"
        exp_prompt_dir = "./gradio_examples/image_prompts/prompts/"
        example = gr.Examples(
            examples=[
                [
                    exp_image_dir + "0.jpeg", exp_prompt_dir + "0.png", "",
                    "none", 0.3, 0.5, 100
                ],
                [
                    exp_image_dir + "1.png", exp_prompt_dir + "1.png", "",
                    "padding", 0.2, 0.1, 100
                ],
                [
                    exp_image_dir + "2.png", exp_prompt_dir + "2.png", "",
                    "padding", 0.0, 0.1, 200
                ],
                [
                    exp_image_dir + "3.png", exp_prompt_dir + "3.png", "",
                    "padding", 0.3, 0.5, 100
                ],
                [
                    exp_image_dir + "4.png", exp_prompt_dir + "4.png", "",
                    "padding", 0.01, 0.1, 200
                ],
                [
                    exp_image_dir + "5.png", exp_prompt_dir + "5.png", "",
                    "none", 0.3, 0.5, 100
                ],
            ],
            inputs=[
                image, prompt_image, input_text, add_padding, score_thr,
                nms_thr, max_num_boxes
            ],
        )

        submit.click(
            partial(run_image, runner, vision_encoder, vision_processor,
                    padding_embed), [
                        image,
                        input_text,
                        prompt_image,
                        add_padding,
                        max_num_boxes,
                        score_thr,
                        nms_thr,
                    ], [output_image])
        clear.click(lambda: [None, None, '', None], None,
                    [image, prompt_image, input_text, output_image])

        demo.launch(server_name='0.0.0.0',
                    server_port=38721)  # port 80 does not work for me


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

    # init vision encoder
    clip_model = "/group/40034/adriancheng/pretrained_models/open-ai-clip-vit-base-patch32"
    vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model)
    processor = AutoProcessor.from_pretrained(clip_model)
    device = 'cuda:0'
    vision_model.to(device)

    texts = [' ']
    tokenizer = AutoTokenizer.from_pretrained(clip_model)
    text_model = CLIPTextModelWithProjection.from_pretrained(clip_model)
    # device = 'cuda:0'
    text_model.to(device)
    texts = tokenizer(text=texts, return_tensors='pt', padding=True)
    texts = texts.to(device)
    text_outputs = text_model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    txt_feats = txt_feats.reshape(-1, txt_feats.shape[-1])
    txt_feats = txt_feats[0].unsqueeze(0)
    demo(runner, args, vision_model, processor, txt_feats)
