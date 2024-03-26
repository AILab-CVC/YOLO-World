# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DeployFocus(nn.Module):

    def __init__(self, orin_Focus: nn.Module):
        super().__init__()
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channel, height, width = x.shape
        x = x.reshape(batch_size, channel, -1, 2, width)
        x = x.reshape(batch_size, channel, x.shape[2], 2, -1, 2)
        half_h = x.shape[2]
        half_w = x.shape[4]
        x = x.permute(0, 5, 3, 1, 2, 4)
        x = x.reshape(batch_size, channel * 4, half_h, half_w)

        return self.conv(x)


class NcnnFocus(nn.Module):

    def __init__(self, orin_Focus: nn.Module):
        super().__init__()
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, f'focus for yolox needs even feature\
            height and width, got {(h, w)}.'

        x = x.reshape(batch_size, c * h, 1, w)
        _b, _c, _h, _w = x.shape
        g = _c // 2
        # fuse to ncnn's shufflechannel
        x = x.view(_b, g, 2, _h, _w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(_b, -1, _h, _w)

        x = x.reshape(_b, c * h * w, 1, 1)

        _b, _c, _h, _w = x.shape
        g = _c // 2
        # fuse to ncnn's shufflechannel
        x = x.view(_b, g, 2, _h, _w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(_b, -1, _h, _w)

        x = x.reshape(_b, c * 4, h // 2, w // 2)

        return self.conv(x)


class GConvFocus(nn.Module):

    def __init__(self, orin_Focus: nn.Module):
        super().__init__()
        device = next(orin_Focus.parameters()).device
        self.weight1 = torch.tensor([[1., 0], [0, 0]]).expand(3, 1, 2,
                                                              2).to(device)
        self.weight2 = torch.tensor([[0, 0], [1., 0]]).expand(3, 1, 2,
                                                              2).to(device)
        self.weight3 = torch.tensor([[0, 1.], [0, 0]]).expand(3, 1, 2,
                                                              2).to(device)
        self.weight4 = torch.tensor([[0, 0], [0, 1.]]).expand(3, 1, 2,
                                                              2).to(device)
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x: Tensor) -> Tensor:
        conv1 = F.conv2d(x, self.weight1, stride=2, groups=3)
        conv2 = F.conv2d(x, self.weight2, stride=2, groups=3)
        conv3 = F.conv2d(x, self.weight3, stride=2, groups=3)
        conv4 = F.conv2d(x, self.weight4, stride=2, groups=3)
        return self.conv(torch.cat([conv1, conv2, conv3, conv4], dim=1))
