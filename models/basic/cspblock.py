import torch
import torch.nn as nn
from .conv import Conv
from .bottleneck import Bottleneck


class CSPBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 nblocks=1,
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(CSPBlock, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.cv3 = Conv(2 * inter_dim, out_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.m = nn.Sequential(*[
            Bottleneck(inter_dim, inter_dim, expand_ratio=1.0, shortcut=shortcut,
                       depthwise=depthwise, norm_type=norm_type, act_type=act_type)
                       for _ in range(nblocks)
                       ])

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x1)
        out = self.cv3(torch.cat([x3, x2], dim=1))

        return out
