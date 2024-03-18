from enum import Enum

import torch
import torch.nn.functional as F


class MMYOLOBackend(Enum):
    AX620A = 'ax620a'
    COREML = 'coreml'
    HORIZONX3 = 'horizonx3'
    NCNN = 'ncnn'
    ONNXRUNTIME = 'onnxruntime'
    OPENVINO = 'openvino'
    PPLNN = 'pplnn'
    RKNN = 'rknn'
    TENSORRT8 = 'tensorrt8'
    TENSORRT7 = 'tensorrt7'
    TORCHSCRIPT = 'torchscript'
    TVM = 'tvm'


def HSigmoid__forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.hardsigmoid(x, inplace=True)
