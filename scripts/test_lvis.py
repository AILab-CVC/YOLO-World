# Copyright (c) Tencent. All rights reserved.
import argparse
import logging
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from torch.distributed import init_process_group

from yolow.data import build_lvis_testloader
from yolow.dist_utils import broadcast, get_rank, sync_random_seed
from yolow.engine.eval import LVISMetric
from yolow.logger import setup_logger
from yolow.model import (build_yolov8_backbone, build_yoloworld_backbone, build_yoloworld_data_preprocessor,
                         build_yoloworld_detector, build_yoloworld_head, build_yoloworld_neck, build_yoloworld_text)
from yolow.model.misc import revert_sync_batchnorm


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World test (and eval)')
    parser.add_argument('model_size', choices=['n', 's', 'm', 'l', 'x', 'xl'], help='model size')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--deterministic', action='store_true', help='Whether to set the deterministic option for CUDNN backend')
    args = parser.parse_args()
    return args


def create_model(args, logger):
    # We have predefined settings in `model/model_cfgs`,
    # including the default architectures for different
    # sizes of YOLO-World models.
    # You can further specify some items via model_args.
    model_args = dict(
        yoloworld_data_preprocessor=dict(),
        # deepen_factor, widen_factor
        yolov8_backbone=dict(),
        yoloworld_text=dict(
            # # avoid downloading
            # model_name='./pretrained/clip-vit-base-patch32'
            # # specify the version of text model when it is different from original config
            # model_name='openai/clip-vit-large-patch14-336',
            # channels=768
        ),
        # with_text_model
        yoloworld_backbone=dict(),
        yoloworld_neck=dict(),
        # use_bn_head
        yoloworld_head_module=dict(),
        # num_train_classes, num_test_classes
        yoloworld_detector=dict(),
    )

    # test build model
    if (get_rank() == 0):
        logger.info(f'Building yolo_world_{args.model_size} model')
    data_preprocessor = build_yoloworld_data_preprocessor(args.model_size, args=model_args)
    yolov8_backbone = build_yolov8_backbone(args.model_size, args=model_args)
    text_backbone = build_yoloworld_text(args.model_size, args=model_args)
    yolow_backbone = build_yoloworld_backbone(args.model_size, yolov8_backbone, text_backbone, args=model_args)
    yolow_neck = build_yoloworld_neck(args.model_size, args=model_args)
    yolow_head = build_yoloworld_head(args.model_size, args=model_args)
    yoloworld_model = build_yoloworld_detector(
        args.model_size, yolow_backbone, yolow_neck, yolow_head, data_preprocessor, args=model_args)

    # test load ckpt (mandatory)
    if (get_rank() == 0):
        logger.info(f'Loading checkpoint from {osp.abspath(args.checkpoint)}')
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    yoloworld_model.load_state_dict(ckpt['state_dict'], strict=True)
    return yoloworld_model


def ddp_setup(args):
    # initialize multi-process and (or) distributed environment.
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Set random seed to guarantee reproducible results.
    seed = sync_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def run_test(args, name, log_period=50):
    # launch distributed process
    ddp_setup(args)

    # setup logger
    logger = logging.getLogger('yolow')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=osp.join(args.work_dir, name, f'{name}.log'))

    # create model
    model = create_model(args, logger)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if (device == 'cpu'):
        model = revert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[int(os.environ['LOCAL_RANK'])],
    )

    # create test dataloader
    # TODO: support more datasets
    if (get_rank() == 0):
        logger.info('Loading LVIS dataloader')
    dataloader = build_lvis_testloader(
        # img_scale=(1280, 1280),
        img_scale=(640, 640),
        val_batch_size_per_gpu=1,
        val_num_workers=4,
        persistent_workers=True)

    # create test evaluator
    ann_file = 'data/coco/lvis/lvis_v1_minival_inserted_image_name.json'
    if (get_rank() == 0):
        logger.info(f'Building LVIS evaluator from {osp.abspath(ann_file)}')
    evaluator = LVISMetric(
        ann_file=ann_file, metric='bbox', format_only=False, outfile_prefix=f'{osp.join(args.work_dir, name)}/results')

    # test loop
    if (get_rank() == 0):
        logger.info(f'Start testing (LEN: {len(dataloader)})')
    model.eval()
    start = time.perf_counter()
    for idx, data in enumerate(dataloader):
        data_time = time.perf_counter() - start
        outputs = model.module.test_step(data)
        evaluator.process(data_samples=outputs, data_batch=data)
        iter_time = time.perf_counter() - start
        if (idx % log_period == 0 and get_rank() == 0):
            logger.info(f'TEST [{idx}/{len(dataloader)}]\t'
                        f'iter_time: {iter_time:.4f}\t'
                        f'data_time: {data_time:.4f}\t'
                        f'memory: {int(torch.cuda.max_memory_allocated()/1024.0/1024.0)}')
        start = time.perf_counter()

    # compute metrics
    metrics = evaluator.evaluate(len(dataloader.dataset))
    if (get_rank() == 0): logger.info(metrics)

    return metrics


def main():
    # parse arguments
    args = parse_args()
    if args.work_dir is None:
        args.work_dir = f'./work_dirs/yolo_world_{args.model_size}_test'

    # create work_dir
    _timestamp = torch.tensor(time.time(), dtype=torch.float64)
    broadcast(_timestamp)  # sync timestamp among different processeses
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(_timestamp.item()))
    os.makedirs(osp.join(osp.expanduser(args.work_dir), timestamp), mode=0o777, exist_ok=True)

    # start testing
    run_test(args, timestamp)


if __name__ == '__main__':
    main()
