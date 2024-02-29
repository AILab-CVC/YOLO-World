# Copyright (c) Tencent Inc. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from functools import partial

import mmengine
import torch.multiprocessing as mp
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import (create_calib_input_data, extract_model,
                           get_predefined_partition_cfg, torch2onnx,
                           torch2torchscript, visualize_model)
from mmdeploy.apis.core import PIPELINE_MANAGER
from mmdeploy.apis.utils import to_backend
from mmdeploy.backend.sdk.export_info import export2SDK
from mmdeploy.utils import (IR, Backend, get_backend, get_calib_filename,
                            get_ir_config, get_partition_config,
                            get_root_logger, load_config, target_wrapper)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to backends.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument('img', help='image used to convert model model')
    parser.add_argument(
        '--test-img',
        default=None,
        type=str,
        nargs='+',
        help='image used to test model')
    parser.add_argument(
        '--work-dir',
        default=os.getcwd(),
        help='the dir to save logs and models')
    parser.add_argument(
        '--calib-dataset-cfg',
        help='dataset config path used to calibrate in int8 mode. If not \
            specified, it will use "val" dataset in model config instead.',
        default=None)
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    parser.add_argument(
        '--show', action='store_true', help='Show detection outputs')
    parser.add_argument(
        '--dump-info', action='store_true', help='Output information for SDK')
    parser.add_argument(
        '--quant-image-dir',
        default=None,
        help='Image directory for quantize model.')
    parser.add_argument(
        '--quant', action='store_true', help='Quantize model to low bit.')
    parser.add_argument(
        '--uri',
        default='192.168.1.1:60000',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')
    args = parser.parse_args()
    return args


def create_process(name, target, args, kwargs, ret_value=None):
    logger = get_root_logger()
    logger.info(f'{name} start.')
    log_level = logger.level

    wrap_func = partial(target_wrapper, target, log_level, ret_value)

    process = Process(target=wrap_func, args=args, kwargs=kwargs)
    process.start()
    process.join()

    if ret_value is not None:
        if ret_value.value != 0:
            logger.error(f'{name} failed.')
            exit(1)
        else:
            logger.info(f'{name} success.')


def torch2ir(ir_type: IR):
    """Return the conversion function from torch to the intermediate
    representation.

    Args:
        ir_type (IR): The type of the intermediate representation.
    """
    if ir_type == IR.ONNX:
        return torch2onnx
    elif ir_type == IR.TORCHSCRIPT:
        return torch2torchscript
    else:
        raise KeyError(f'Unexpected IR type {ir_type}')


def main():
    args = parse_args()
    set_start_method('spawn', force=True)
    logger = get_root_logger()
    log_level = logging.getLevelName(args.log_level)
    logger.setLevel(log_level)

    pipeline_funcs = [
        torch2onnx, torch2torchscript, extract_model, create_calib_input_data
    ]
    PIPELINE_MANAGER.enable_multiprocess(True, pipeline_funcs)
    PIPELINE_MANAGER.set_log_level(log_level, pipeline_funcs)

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint
    quant = args.quant
    quant_image_dir = args.quant_image_dir

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # create work_dir if not
    mmengine.mkdir_or_exist(osp.abspath(args.work_dir))

    if args.dump_info:
        export2SDK(
            deploy_cfg,
            model_cfg,
            args.work_dir,
            pth=checkpoint_path,
            device=args.device)

    ret_value = mp.Value('d', 0, lock=False)

    # convert to IR
    ir_config = get_ir_config(deploy_cfg)
    ir_save_file = ir_config['save_file']
    ir_type = IR.get(ir_config['type'])
    torch2ir(ir_type)(
        args.img,
        args.work_dir,
        ir_save_file,
        deploy_cfg_path,
        model_cfg_path,
        checkpoint_path,
        device=args.device)

    # convert backend
    ir_files = [osp.join(args.work_dir, ir_save_file)]

    # partition model
    partition_cfgs = get_partition_config(deploy_cfg)

    if partition_cfgs is not None:

        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_ir_file = ir_files[0]
        ir_files = []
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            extract_model(
                origin_ir_file,
                start,
                end,
                dynamic_axes=dynamic_axes,
                save_file=save_path)

            ir_files.append(save_path)

    # calib data
    calib_filename = get_calib_filename(deploy_cfg)
    if calib_filename is not None:
        calib_path = osp.join(args.work_dir, calib_filename)
        create_calib_input_data(
            calib_path,
            deploy_cfg_path,
            model_cfg_path,
            checkpoint_path,
            dataset_cfg=args.calib_dataset_cfg,
            dataset_type='val',
            device=args.device)

    backend_files = ir_files
    # convert backend
    backend = get_backend(deploy_cfg)

    # preprocess deploy_cfg
    if backend == Backend.RKNN:
        # TODO: Add this to task_processor in the future
        import tempfile

        from mmdeploy.utils import (get_common_config, get_normalization,
                                    get_quantization_config,
                                    get_rknn_quantization)
        quantization_cfg = get_quantization_config(deploy_cfg)
        common_params = get_common_config(deploy_cfg)
        if get_rknn_quantization(deploy_cfg) is True:
            transform = get_normalization(model_cfg)
            common_params.update(
                dict(
                    mean_values=[transform['mean']],
                    std_values=[transform['std']]))

        dataset_file = tempfile.NamedTemporaryFile(suffix='.txt').name
        with open(dataset_file, 'w') as f:
            f.writelines([osp.abspath(args.img)])
        if quantization_cfg.get('dataset', None) is None:
            quantization_cfg['dataset'] = dataset_file
    if backend == Backend.ASCEND:
        # TODO: Add this to backend manager in the future
        if args.dump_info:
            from mmdeploy.backend.ascend import update_sdk_pipeline
            update_sdk_pipeline(args.work_dir)

    if backend == Backend.VACC:
        # TODO: Add this to task_processor in the future

        from onnx2vacc_quant_dataset import get_quant

        from mmdeploy.utils import get_model_inputs

        deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
        model_inputs = get_model_inputs(deploy_cfg)

        for onnx_path, model_input in zip(ir_files, model_inputs):

            quant_mode = model_input.get('qconfig', {}).get('dtype', 'fp16')
            assert quant_mode in ['int8',
                                  'fp16'], quant_mode + ' not support now'
            shape_dict = model_input.get('shape', {})

            if quant_mode == 'int8':
                create_process(
                    'vacc quant dataset',
                    target=get_quant,
                    args=(deploy_cfg, model_cfg, shape_dict, checkpoint_path,
                          args.work_dir, args.device),
                    kwargs=dict(),
                    ret_value=ret_value)

    # convert to backend
    PIPELINE_MANAGER.set_log_level(log_level, [to_backend])
    if backend == Backend.TENSORRT:
        PIPELINE_MANAGER.enable_multiprocess(True, [to_backend])
    backend_files = to_backend(
        backend,
        ir_files,
        work_dir=args.work_dir,
        deploy_cfg=deploy_cfg,
        log_level=log_level,
        device=args.device,
        uri=args.uri)

    # ncnn quantization
    if backend == Backend.NCNN and quant:
        from onnx2ncnn_quant_table import get_table

        from mmdeploy.apis.ncnn import get_quant_model_file, ncnn2int8
        model_param_paths = backend_files[::2]
        model_bin_paths = backend_files[1::2]
        backend_files = []
        for onnx_path, model_param_path, model_bin_path in zip(
                ir_files, model_param_paths, model_bin_paths):

            deploy_cfg, model_cfg = load_config(deploy_cfg_path,
                                                model_cfg_path)
            quant_onnx, quant_table, quant_param, quant_bin = get_quant_model_file(  # noqa: E501
                onnx_path, args.work_dir)

            create_process(
                'ncnn quant table',
                target=get_table,
                args=(onnx_path, deploy_cfg, model_cfg, quant_onnx,
                      quant_table, quant_image_dir, args.device),
                kwargs=dict(),
                ret_value=ret_value)

            create_process(
                'ncnn_int8',
                target=ncnn2int8,
                args=(model_param_path, model_bin_path, quant_table,
                      quant_param, quant_bin),
                kwargs=dict(),
                ret_value=ret_value)
            backend_files += [quant_param, quant_bin]

    if args.test_img is None:
        args.test_img = args.img

    extra = dict(
        backend=backend,
        output_file=osp.join(args.work_dir, f'output_{backend.value}.jpg'),
        show_result=args.show)
    if backend == Backend.SNPE:
        extra['uri'] = args.uri

    # get backend inference result, try render
    create_process(
        f'visualize {backend.value} model',
        target=visualize_model,
        args=(model_cfg_path, deploy_cfg_path, backend_files, args.test_img,
              args.device),
        kwargs=extra,
        ret_value=ret_value)

    # get pytorch model inference result, try visualize if possible
    create_process(
        'visualize pytorch model',
        target=visualize_model,
        args=(model_cfg_path, deploy_cfg_path, [checkpoint_path],
              args.test_img, args.device),
        kwargs=dict(
            backend=Backend.PYTORCH,
            output_file=osp.join(args.work_dir, 'output_pytorch.jpg'),
            show_result=args.show),
        ret_value=ret_value)
    logger.info('All process success.')


if __name__ == '__main__':
    main()
