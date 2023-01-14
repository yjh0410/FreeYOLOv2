import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.conv import Conv
from ..basic.csp_elanblock import CSP_ELANBlock


# DownSample
class DownSample(nn.Module):
    def __init__(self, in_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # maxpooling branch
        inter_dim = in_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        # conv-s2 branch
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
        )


    def forward(self, x):
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


# CSP-ELAN-PaFPN
class CSP_ELAN_PaFPN(nn.Module):
    def __init__(self, 
                 width=1.0,
                 depth=1.0, 
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(CSP_ELAN_PaFPN, self).__init__()
        # --------- Basic Parameters ----------
        ## base dims
        self.in_dims = [512, 1024, 1024]
        self.out_dim = 256
        ## scaled dims
        self.scaled_in_dims = [int(width * dim) for dim in self.in_dims]
        self.scaled_out_dim = int(width * self.out_dim)

        # --------- Network Parameters ----------
        c3, c4, c5 = self.scaled_in_dims

        self.head_conv_0 = Conv(c5, c5//2, k=1, norm_type=norm_type, act_type=act_type)  # 10
        self.head_csp_0 = CSP_ELANBlock(
            c4 + c5//2, c4, depth=int(3*depth), act_type=act_type,
            norm_type=norm_type, shortcut=False, depthwise=depthwise)

        # P3/8-small
        self.head_conv_1 = Conv(c4, c4//2, k=1, norm_type=norm_type, act_type=act_type)  # 14
        self.head_csp_1 = CSP_ELANBlock(
            c3 + c4//2, c3, depth=int(3*depth), act_type=act_type,
            norm_type=norm_type, shortcut=False, depthwise=depthwise)

        # P4/16-medium
        self.head_conv_2 = DownSample(c3, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_csp_2 = CSP_ELANBlock(
            c3 + c4//2, c4, depth=int(3*depth), act_type=act_type,
            norm_type=norm_type, shortcut=False, depthwise=depthwise)

        # P8/32-large
        self.head_conv_3 = DownSample(c4, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_csp_3 = CSP_ELANBlock(
            c4 + c5//2, c5, depth=int(3*depth), act_type=act_type,
            norm_type=norm_type, shortcut=False, depthwise=depthwise)

        # output proj layers
        self.out_layers = nn.ModuleList([
            Conv(dim, self.scaled_out_dim, k=1,
                    act_type=act_type, norm_type=norm_type)
                    for dim in self.scaled_in_dims
                    ])

    def forward(self, features):
        c3, c4, c5 = features

        c6 = self.head_conv_0(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)   # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        # P3/8
        c10 = self.head_conv_1(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)   # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)  # to det
        # p4/16
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)  # to det
        # p5/32
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)  # to det

        out_feats = [c13, c16, c19] # [P3, P4, P5]

        # output proj layers
        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
            
        return out_feats_proj
