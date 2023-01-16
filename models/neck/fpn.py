import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.conv import Conv, ConvBlocks
from .spp import SPPFBlock

class UpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return F.interpolate(input=x, 
                             size=self.size, 
                             scale_factor=self.scale_factor, 
                             mode=self.mode, 
                             align_corners=self.align_corner
                                               )

# BasicFPN
class BasicFPN(nn.Module):
    def __init__(self,
                 in_dims=[256, 512, 1024],
                 out_dim=None,
                 act_type='silu',
                 norm_type='BN',
                 spp_block=False):
        super(BasicFPN, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims

        # P5 -> P4
        if spp_block:
            self.head_convblock_0 = SPPFBlock(c5, c5//2, pooling_size=5, act_type=act_type, norm_type=norm_type)
        else:
            self.head_convblock_0 = ConvBlocks(c5, c5//2, act_type=act_type, norm_type=norm_type)
        self.head_conv_0 = Conv(c5//2, c4//2, k=1, act_type=act_type, norm_type=norm_type)
        self.head_conv_1 = Conv(c5//2, c5, k=3, p=1, act_type=act_type, norm_type=norm_type)

        # P4 -> P3
        self.head_convblock_1 = ConvBlocks(c4 + c4//2, c4//2, act_type=act_type, norm_type=norm_type)
        self.head_conv_2 = Conv(c4//2, c3//2, k=1, act_type=act_type, norm_type=norm_type)
        self.head_conv_3 = Conv(c4//2, c4, k=3, p=1, act_type=act_type, norm_type=norm_type)

        # P3
        self.head_convblock_2 = ConvBlocks(c3 + c3//2, c3//2, act_type=act_type, norm_type=norm_type)
        self.head_conv_4 = Conv(c3//2, c3, k=3, p=1, act_type=act_type, norm_type=norm_type)

        # output proj layers
        if self.out_dim is not None:
            self.out_layers = nn.ModuleList([
                Conv(in_dim, self.out_dim, k=1,
                        norm_type=norm_type, act_type=act_type)
                        for in_dim in in_dims
                        ])
        else:
            self.out_dim = [c3, c4, c5]


    def forward(self, features):
        c3, c4, c5 = features
        
        # p5/32
        p5 = self.head_convblock_0(c5)
        p5_up = F.interpolate(self.head_conv_0(p5), scale_factor=2.0)
        p5 = self.head_conv_1(p5)

        # p4/16
        p4 = self.head_convblock_1(torch.cat([c4, p5_up], dim=1))
        p4_up = F.interpolate(self.head_conv_2(p4), scale_factor=2.0)
        p4 = self.head_conv_3(p4)

        # P3/8
        p3 = self.head_convblock_2(torch.cat([c3, p4_up], dim=1))
        p3 = self.head_conv_4(p3)

        out_feats = [p3, p4, p5]

        # output proj layers
        if self.out_dim is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))

            return out_feats_proj

        return out_feats
