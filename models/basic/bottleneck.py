import torch
import torch.nn as nn
from .conv import Conv


class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 kernel_size=(3, 3),
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(Bottleneck, self).__init__()
        inter_dim = int(in_dim * expand_ratio)  # hidden channels       
        self.cv1 = Conv(in_dim, inter_dim, k=kernel_size[0], p=kernel_size[0]//2,
                        norm_type=norm_type, act_type=act_type, depthwise=depthwise if kernel_size[0] > 1 else False)
        self.cv2 = Conv(inter_dim, out_dim, k=kernel_size[1], p=kernel_size[1]//2,
                        norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h
