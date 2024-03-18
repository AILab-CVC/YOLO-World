import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    import tensorrt as trt
except Exception:
    trt = None
import warnings

import numpy as np
import torch

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


class EngineBuilder:

    def __init__(
            self,
            checkpoint: Union[str, Path],
            opt_shape: Union[Tuple, List] = (1, 3, 640, 640),
            device: Optional[Union[str, int, torch.device]] = None) -> None:
        checkpoint = Path(checkpoint) if isinstance(checkpoint,
                                                    str) else checkpoint
        assert checkpoint.exists() and checkpoint.suffix == '.onnx'
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')

        self.checkpoint = checkpoint
        self.opt_shape = np.array(opt_shape, dtype=np.float32)
        self.device = device

    def __build_engine(self,
                       scale: Optional[List[List]] = None,
                       fp16: bool = True,
                       with_profiling: bool = True) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = torch.cuda.get_device_properties(
            self.device).total_memory
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(self.checkpoint)):
            raise RuntimeError(
                f'failed to load ONNX file: {str(self.checkpoint)}')
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        profile = None
        dshape = -1 in network.get_input(0).shape
        if dshape:
            profile = builder.create_optimization_profile()
            if scale is None:
                scale = np.array(
                    [[1, 1, 0.5, 0.5], [1, 1, 1, 1], [4, 1, 1.5, 1.5]],
                    dtype=np.float32)
                scale = (self.opt_shape * scale).astype(np.int32)
            elif isinstance(scale, List):
                scale = np.array(scale, dtype=np.int32)
                assert scale.shape[0] == 3, 'Input a wrong scale list'
            else:
                raise NotImplementedError

        for inp in inputs:
            logger.log(
                trt.Logger.WARNING,
                f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
            if dshape:
                profile.set_shape(inp.name, *scale)
        for out in outputs:
            logger.log(
                trt.Logger.WARNING,
                f'output "{out.name}" with shape{out.shape} {out.dtype}')
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        self.weight = self.checkpoint.with_suffix('.engine')
        if dshape:
            config.add_optimization_profile(profile)
        if with_profiling:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        with builder.build_engine(network, config) as engine:
            self.weight.write_bytes(engine.serialize())
        logger.log(
            trt.Logger.WARNING, f'Build tensorrt engine finish.\n'
            f'Save in {str(self.weight.absolute())}')

    def build(self,
              scale: Optional[List[List]] = None,
              fp16: bool = True,
              with_profiling=True):
        self.__build_engine(scale, fp16, with_profiling)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='Image size of height and width')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='TensorRT builder device')
    parser.add_argument(
        '--scales',
        type=str,
        default='[[1,3,640,640],[1,3,640,640],[1,3,640,640]]',
        help='Input scales for build dynamic input shape engine')
    parser.add_argument(
        '--fp16', action='store_true', help='Build model with fp16 mode')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1
    return args


def main(args):
    img_size = (1, 3, *args.img_size)
    try:
        scales = eval(args.scales)
    except Exception:
        print('Input scales is not a python variable')
        print('Set scales default None')
        scales = None
    builder = EngineBuilder(args.checkpoint, img_size, args.device)
    builder.build(scales, fp16=args.fp16)


if __name__ == '__main__':
    args = parse_args()
    main(args)
