import torch
import torch.nn as nn
from .conv import Conv
from .bottleneck import Bottleneck


# ELAN-CSP-Block
class ELAN_CSP_Block(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 nblocks=1,
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(ELAN_CSP_Block, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.m = nn.Sequential(*(
            Bottleneck(inter_dim, inter_dim, 1.0, shortcut, depthwise, act_type, norm_type)
            for _ in range(nblocks)))
        self.cv3 = Conv((2 + nblocks) * inter_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        out = list([x1, x2])

        out.extend(m(out[-1]) for m in self.m)

        out = self.cv3(torch.cat(out, dim=1))

        return out


# DownSample
class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        inter_dim = out_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out
