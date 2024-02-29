# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from copy import deepcopy

from mmengine import DictAction

from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import load_config
from mmdeploy.utils.timer import TimeCounter


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDeploy test (and eval) a backend.')
    parser.add_argument('deploy_cfg', help='Deploy config path')
    parser.add_argument('model_cfg', help='Model config path')
    parser.add_argument(
        '--model', type=str, nargs='+', help='Input model files.')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--work-dir',
        default='./work_dir',
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
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
    parser.add_argument(
        '--log2file',
        type=str,
        help='log evaluation results and speed to file',
        default=None)
    parser.add_argument(
        '--speed-test', action='store_true', help='activate speed test')
    parser.add_argument(
        '--warmup',
        type=int,
        help='warmup before counting inference elapse, require setting '
        'speed-test first',
        default=10)
    parser.add_argument(
        '--log-interval',
        type=int,
        help='the interval between each log, require setting '
        'speed-test first',
        default=100)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='the batch size for test, would override `samples_per_gpu`'
        'in  data config.')
    parser.add_argument(
        '--uri',
        action='store_true',
        default='192.168.1.1:60000',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        work_dir = args.work_dir
    elif model_cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])

    # merge options for model cfg
    if args.cfg_options is not None:
        model_cfg.merge_from_dict(args.cfg_options)

    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)

    # prepare the dataset loader
    test_dataloader = deepcopy(model_cfg['test_dataloader'])
    if isinstance(test_dataloader, list):
        dataset = []
        for loader in test_dataloader:
            ds = task_processor.build_dataset(loader['dataset'])
            dataset.append(ds)
            loader['dataset'] = ds
            loader['batch_size'] = args.batch_size
            loader = task_processor.build_dataloader(loader)
        dataloader = test_dataloader
    else:
        test_dataloader['batch_size'] = args.batch_size
        dataset = task_processor.build_dataset(test_dataloader['dataset'])
        test_dataloader['dataset'] = dataset
        dataloader = task_processor.build_dataloader(test_dataloader)

    # load the model of the backend
    model = task_processor.build_backend_model(
        args.model,
        data_preprocessor_updater=task_processor.update_data_preprocessor)
    destroy_model = model.destroy
    is_device_cpu = (args.device == 'cpu')

    runner = task_processor.build_test_runner(
        model,
        work_dir,
        log_file=args.log2file,
        show=args.show,
        show_dir=args.show_dir,
        wait_time=args.wait_time,
        interval=args.interval,
        dataloader=dataloader)

    if args.speed_test:
        with_sync = not is_device_cpu

        with TimeCounter.activate(
                warmup=args.warmup,
                log_interval=args.log_interval,
                with_sync=with_sync,
                file=args.log2file,
                batch_size=args.batch_size):
            runner.test()

    else:
        runner.test()
    # only effective when the backend requires explicit clean-up (e.g. Ascend)
    destroy_model()


if __name__ == '__main__':
    main()
