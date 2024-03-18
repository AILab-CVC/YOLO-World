import torch
import torch.nn as nn
from torch import Tensor


class DeployC2f(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x_main = self.main_conv(x)
        x_main = [x_main, x_main[:, self.mid_channels:, ...]]
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.pop(1)
        return self.final_conv(torch.cat(x_main, 1))
