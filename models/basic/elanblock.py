import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.conv import Conv


# ELANBlock
class ELANBlock(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANBlock, self).__init__()
        if isinstance(expand_ratio, float):
            inter_dim = int(in_dim * expand_ratio)
            inter_dim2 = inter_dim
        elif isinstance(expand_ratio, list):
            assert len(expand_ratio) == 2
            e1, e2 = expand_ratio
            inter_dim = int(in_dim * e1)
            inter_dim2 = int(inter_dim * e2)
        # branch-1
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        # branch-2
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        # branch-3
        for idx in range(round(3*depth)):
            if idx == 0:
                cv3 = [Conv(inter_dim, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            else:
                cv3.append(Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
        self.cv3 = nn.Sequential(*cv3)
        # branch-4
        self.cv4 = nn.Sequential(*[
            Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(round(3*depth))
        ])
        # output
        self.out = Conv(inter_dim*2 + inter_dim2*2, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

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
