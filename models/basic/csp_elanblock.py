import torch
import torch.nn as nn
from .conv import Conv


# Bottleneck
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 act_type='silu',
                 norm_type='BN',
                 shortcut=False,
                 depthwise=False):
        super().__init__()
        self.cv1 = Conv(in_dim, in_dim, k=3, p=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, in_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.shortcut = shortcut

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h


# CSP-ELANBlock - same to 'C2f' proposed by YOLOv8
class CSP_ELANBlock(nn.Module): 
    def __init__(self,
                 in_dim,
                 out_dim,
                 depth=1,
                 act_type='silu',
                 norm_type='BN',
                 shortcut=False,
                 depthwise=False):
        super(CSP_ELANBlock, self).__init__()
        inter_dim = out_dim // 4
        self.cv1 = Conv(in_dim, 2*inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(*[
            Bottleneck(inter_dim, act_type=act_type, norm_type=norm_type,
                       shortcut=shortcut, depthwise=depthwise)
            for _ in range(depth)
            ])
        self.cv3 = nn.Sequential(*[
            Bottleneck(inter_dim, act_type=act_type, norm_type=norm_type,
                       shortcut=shortcut, depthwise=depthwise)
            for _ in range(depth)
            ])
        self.cv4 = Conv(inter_dim*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1, x2 = self.cv1(x).chunk(2, 1)
        x3 = self.cv2(x2)
        x4 = self.cv3(x3)
        y = torch.cat([x1, x2, x3, x4], dim=1)

        return self.cv4(y)

