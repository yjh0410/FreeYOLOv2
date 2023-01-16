import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.conv import Conv
from ..basic.elanblock import ELANBlock, DownSample


# PaFPN-ELAN (YOLOv7's)
class ELAN_PaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[512, 1024, 512],
                 out_dim=None,
                 width=1.0,
                 depth=1.0,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(ELAN_PaFPN, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        c3, c4, c5 = in_dims

        # top dwon
        ## P5 -> P4
        self.cv1 = Conv(c5, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(c4, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_1 = ELANBlock(in_dim=int(256 * width) + int(256 * width),
                                     out_dim=int(256 * width),
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P3
        self.cv3 = Conv(int(256 * width), int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv4 = Conv(c3, int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_2 = ELANBlock(in_dim=int(128 * width) + int(128 * width),
                                     out_dim=int(128 * width),  # 128
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # bottom up
        # P3 -> P4
        self.mp1 = DownSample(int(128 * width), out_dim=int(256 * width), 
                              act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_3 = ELANBlock(in_dim=int(256 * width) + int(256 * width),
                                     out_dim=int(256 * width),  # 256
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P5
        self.mp2 = DownSample(int(256 * width), out_dim=int(512 * width),
                              act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_4 = ELANBlock(in_dim=int(512 * width) + c5,
                                     out_dim=int(512 * width),  # 512
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # head conv
        self.head_conv_1 = Conv(int(128 * width), int(256 * width), k=3, p=1,
                                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_conv_2 = Conv(int(256 * width), int(512 * width), k=3, p=1,
                                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_conv_3 = Conv(int(512 * width), int(1024 * width), k=3, p=1,
                                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)

        if self.out_dim is not None:
            # output proj layers
            self.out_layers = nn.ModuleList([
                Conv(in_dim, self.out_dim, k=1,
                        norm_type=norm_type, act_type=act_type)
                        for in_dim in [int(256 * width), int(512 * width), int(1024 * width)]
                        ])
        else:
            self.out_dim = [int(256 * width), int(512 * width), int(1024 * width)]


    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)
        ## P4 -> P3
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.mp1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.head_elan_3(c15)
        # P4 -> P5
        c17 = self.mp2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.head_elan_4(c18)

        c20 = self.head_conv_1(c13)
        c21 = self.head_conv_2(c16)
        c22 = self.head_conv_3(c19)

        out_feats = [c20, c21, c22] # [P3, P4, P5]
        
        if self.out_dim is not None:
            # output proj layers
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats
        