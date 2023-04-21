import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo_free_v2_basic import (
    Conv, RepConv, build_reduce_layer, build_downsample_layer, 
    build_fpn_block, build_fpn_head_conv)


# PaFPN-ELAN (YOLOv7's)
class Yolov7PaFPN(nn.Module):
    def __init__(self,
                 cfg,
                 in_dims=[512, 1024, 512],
                 out_dim=None,
                 width=1.0,
                 deploy=False):
        super(Yolov7PaFPN, self).__init__()
        self.in_dims = in_dims
        c3, c4, c5 = in_dims

        # top dwon
        ## P5 -> P4
        self.reduce_layer_1 = build_reduce_layer(cfg, c5, round(256*width))
        self.reduce_layer_2 = build_reduce_layer(cfg, c4, round(256*width))
        self.top_down_layer_1 = build_fpn_block(cfg, round(256*width) + round(256*width), round(256*width))

        # P4 -> P3
        self.reduce_layer_3 = build_reduce_layer(cfg, round(256*width), round(128*width))
        self.reduce_layer_4 = build_reduce_layer(cfg, c3, round(128*width))
        self.top_down_layer_2 = build_fpn_block(cfg, round(128*width) + round(128*width), round(128*width))

        # bottom up
        # P3 -> P4
        self.downsample_layer_1 = build_downsample_layer(cfg, round(128*width), round(256*width))
        self.bottom_up_layer_1 = build_fpn_block(cfg, round(256*width) + round(256*width), round(256*width))

        # P4 -> P5
        self.downsample_layer_2 = build_downsample_layer(cfg, round(256*width), round(512*width))
        self.bottom_up_layer_2 = build_fpn_block(cfg, round(512*width) + round(512*width), round(512*width))
        
        # Head Conv        
        self.head_conv_1 = build_fpn_head_conv(cfg, round(128*width), round(256*width), deploy)
        self.head_conv_2 = build_fpn_head_conv(cfg, round(256*width), round(512*width), deploy)
        self.head_conv_3 = build_fpn_head_conv(cfg, round(512*width), round(1024*width), deploy)
        
        # output proj layers
        if out_dim is not None:
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1,
                     act_type=cfg['fpn_act'], norm_type=cfg['fpn_norm'])
                     for in_dim in [round(256*width), round(512*width), round(1024*width)]
                     ])
            self.out_dim = [out_dim] * 3
        else:
            self.out_layers = None
            self.out_dim = [round(256*width), round(512*width), round(1024*width)]

        # Fuse RepConv
        if deploy:
            self.fuse_repconv()


    def fuse_repconv(self):
        print('Fusing RepConv layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()


    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = self.reduce_layer_1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.reduce_layer_2(c4)], dim=1)
        c9 = self.top_down_layer_1(c8)
        ## P4 -> P3
        c10 = self.reduce_layer_3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.reduce_layer_4(c3)], dim=1)
        c13 = self.top_down_layer_2(c12)

        # Bottom up
        ## p3 -> P4
        c14 = self.downsample_layer_1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.bottom_up_layer_1(c15)
        ## P4 -> P5
        c17 = self.downsample_layer_2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.bottom_up_layer_2(c18)

        # Head conv
        c20 = self.head_conv_1(c13)
        c21 = self.head_conv_2(c16)
        c22 = self.head_conv_3(c19)
        out_feats = [c20, c21, c22] # [P3, P4, P5]
        
        # output proj layers
        if self.out_layers is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(cfg, in_dims, out_dim=None, deploy=False):
    model = cfg['fpn']
    # build pafpn
    if model == 'yolov7_pafpn':
        fpn_net = Yolov7PaFPN(cfg = cfg,
                              in_dims = in_dims,
                              out_dim = out_dim,
                              width = cfg['width'],
                              deploy = deploy
                              )


    return fpn_net