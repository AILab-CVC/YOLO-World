# Copyright (c) Tencent Inc. All rights reserved.
# This file is modifef from mmyolo/demo/video_demo.py
import argparse

import cv2
import mmcv
import torch
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmengine.utils import track_iter_progress

from mmyolo.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World video demo')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('video', help='video file path')
    parser.add_argument(
        'text',
        help=
        'text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference')
    parser.add_argument('--score-thr',
                        default=0.1,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--out', type=str, help='output video file')
    args = parser.parse_args()
    return args


def inference_detector(model, image, texts, test_pipeline, score_thr=0.3):
    data_info = dict(img_id=0, img=image, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() >
                                        score_thr]
    output.pred_instances = pred_instances
    return output


def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)

    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]

    # reparameterize texts
    model.reparameterize(texts)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in track_iter_progress(video_reader):
        result = inference_detector(model,
                                    frame,
                                    texts,
                                    test_pipeline,
                                    score_thr=args.score_thr)
        visualizer.add_datasample(name='video',
                                  image=frame,
                                  data_sample=result,
                                  draw_gt=False,
                                  show=False,
                                  pred_score_thr=args.score_thr)
        frame = visualizer.get_image()

        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()


if __name__ == '__main__':
    main()
