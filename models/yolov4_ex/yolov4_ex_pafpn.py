import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolov4_ex_basic import Conv, CSPBlock


# PaFPN-CSP
class CSP_PaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024],
                 out_dim=256,
                 width=1.0,
                 depth=1.0,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(CSP_PaFPN, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims

        self.head_conv_0 = Conv(c5, int(512*width), k=1, norm_type=norm_type, act_type=act_type)  # 10
        self.head_csp_0 = CSPBlock(c4 + int(512*width), int(512*width), expand_ratio=0.5, kernel=[1, 3],
                                   nblocks=int(3*depth), shortcut=False, depthwise=depthwise,
                                   norm_type=norm_type, act_type=act_type)

        # P3/8-small
        self.head_conv_1 = Conv(c4, int(256*width), k=1, norm_type=norm_type, act_type=act_type)  # 14
        self.head_csp_1 = CSPBlock(c3 + int(256*width), int(256*width), expand_ratio=0.5, kernel=[1, 3],
                                   nblocks=int(3*depth), shortcut=False, depthwise=depthwise,
                                   norm_type=norm_type, act_type=act_type)

        # P4/16-medium
        self.head_conv_2 = Conv(int(256*width), int(256*width), k=3, p=1, s=2, depthwise=depthwise, norm_type=norm_type, act_type=act_type)
        self.head_csp_2 = CSPBlock(int(256*width) + int(256*width), int(512*width), expand_ratio=0.5, kernel=[1, 3],
                                   nblocks=int(3*depth), shortcut=False, depthwise=depthwise,
                                   norm_type=norm_type, act_type=act_type)

        # P8/32-large
        self.head_conv_3 = Conv(int(512*width), int(512*width), k=3, p=1, s=2, depthwise=depthwise, norm_type=norm_type, act_type=act_type)
        self.head_csp_3 = CSPBlock(int(512*width) + int(512*width), int(1024*width), expand_ratio=0.5, kernel=[1, 3],
                                   nblocks=int(3*depth), shortcut=False, depthwise=depthwise,
                                   norm_type=norm_type, act_type=act_type)

        # output proj layers
        if out_dim is not None:
            # output proj layers
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1,
                        norm_type=norm_type, act_type=act_type)
                        for in_dim in [int(256 * width), int(512 * width), int(1024 * width)]
                        ])
            self.out_dim = [out_dim] * 3

        else:
            self.out_layers = None
            self.out_dim = [int(256 * width), int(512 * width), int(1024 * width)]


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
        if self.out_layers is not None:
            # output proj layers
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(cfg, in_dims, out_dim=None):
    model = cfg['fpn']
    # build neck
    if model == 'csp_pafpn':
        fpn_net = CSP_PaFPN(in_dims=in_dims,
                             out_dim=out_dim,
                             width=cfg['width'],
                             depth=cfg['depth'],
                             act_type=cfg['fpn_act'],
                             norm_type=cfg['fpn_norm'],
                             depthwise=cfg['fpn_depthwise']
                             )


    return fpn_net

