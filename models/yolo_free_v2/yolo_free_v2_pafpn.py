import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolo_free_v2_basic import Conv, ELANBlock, DownSample


# PaFPN-ELAN
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
        print('==============================')
        print('FPN: {}'.format("ELAN_PaFPN"))

        self.in_dims = in_dims
        self.width = width
        self.depth = depth
        c3, c4, c5 = in_dims

        # top dwon
        ## P5 -> P4
        self.cv1 = Conv(c5, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(c4, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_1 = ELANBlock(in_dim=int(256 * width) + int(256 * width),
                                     out_dim=int(256 * width),
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P3
        self.cv3 = Conv(int(256 * width), int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv4 = Conv(c3, int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_2 = ELANBlock(in_dim=int(128 * width) + int(128 * width),
                                     out_dim=int(128 * width),  # 128
                                     expand_ratio=[0.5, 0.5],
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
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P5
        self.mp2 = DownSample(int(256 * width), out_dim=int(512 * width),
                              act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_4 = ELANBlock(in_dim=int(512 * width) + c5,
                                     out_dim=int(512 * width),  # 512
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        if out_dim is not None:
            # output proj layers
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1,
                        norm_type=norm_type, act_type=act_type)
                        for in_dim in [int(128 * width), int(256 * width), int(512 * width)]
                        ])
            self.out_dim = [out_dim] * 3

        else:
            self.out_layers = None
            self.out_dim = [int(128 * width), int(256 * width), int(512 * width)]


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

        out_feats = [c13, c16, c19] # [P3, P4, P5]
        
        if self.out_layers is not None:
            # output proj layers
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


# PaFPN-ELAN-P6
class ELAN_PaFPN_P6(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 768, 512],
                 out_dim=None,
                 width=1.0,
                 depth=1.0,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(ELAN_PaFPN_P6, self).__init__()
        print('==============================')
        print('FPN: {}'.format("ELAN_PaFPN-P6"))

        self.in_dims = in_dims
        self.width = width
        self.depth = depth
        c3, c4, c5, c6 = in_dims

        # top dwon
        ## P6 -> P5
        self.cv0 = Conv(c6, int(384 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv1 = Conv(c5, int(384 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_0 = ELANBlock(in_dim=int(384 * width) + int(384 * width),
                                     out_dim=int(384 * width),
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        ## P5 -> P4
        self.cv2 = Conv(int(384 * width), int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv3 = Conv(c4, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_1 = ELANBlock(in_dim=int(256 * width) + int(256 * width),
                                     out_dim=int(256 * width),
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P3
        self.cv4 = Conv(int(256 * width), int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv5 = Conv(c3, int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_2 = ELANBlock(in_dim=int(128 * width) + int(128 * width),
                                     out_dim=int(128 * width),  # 128
                                     expand_ratio=[0.5, 0.5],
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
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P5
        self.mp2 = DownSample(int(256 * width), out_dim=int(384 * width),
                              act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_4 = ELANBlock(in_dim=int(384 * width) + int(384 * width),
                                     out_dim=int(384 * width),  # 384
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)
        # P5 -> P6
        self.mp3 = DownSample(int(384 * width), out_dim=int(512 * width),
                              act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_5 = ELANBlock(in_dim=int(512 * width) + c6,
                                     out_dim=int(512 * width),  # 512
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        if out_dim is not None:
            # output proj layers
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1,
                        norm_type=norm_type, act_type=act_type)
                        for in_dim in [int(128 * width), int(256 * width), int(384 * width), int(512 * width)]
                        ])
            self.out_dim = [out_dim] * 4

        else:
            self.out_layers = None
            self.out_dim = [int(128 * width), int(256 * width), int(384 * width), int(512 * width)]


    def forward(self, features):
        c3, c4, c5, c6 = features

        # Top down
        ## P6 -> P5
        c7 = self.cv0(c6)
        c8 = F.interpolate(c7, scale_factor=2.0)
        c9 = torch.cat([c8, self.cv1(c5)], dim=1)
        c10 = self.head_elan_0(c9)
        ## P5 -> P4
        c11 = self.cv2(c10)
        c12 = F.interpolate(c11, scale_factor=2.0)
        c13 = torch.cat([c12, self.cv3(c4)], dim=1)
        c14 = self.head_elan_1(c13)
        ## P4 -> P3
        c15 = self.cv4(c14)
        c16 = F.interpolate(c15, scale_factor=2.0)
        c17 = torch.cat([c16, self.cv5(c3)], dim=1)
        c18 = self.head_elan_2(c17)

        # Bottom up
        # p3 -> P4
        c19 = self.mp1(c18)
        c20 = torch.cat([c19, c14], dim=1)
        c21 = self.head_elan_3(c20)
        # P4 -> P5
        c22 = self.mp2(c21)
        c23 = torch.cat([c22, c10], dim=1)
        c24 = self.head_elan_4(c23)
        # P5 -> P6
        c25 = self.mp3(c24)
        c26 = torch.cat([c25, c6], dim=1)
        c27 = self.head_elan_5(c26)

        out_feats = [c18, c21, c24, c27] # [P3, P4, P5, P6]
        
        if self.out_layers is not None:
            # output proj layers
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


# PaFPN-ELAN-P7
class ELAN_PaFPN_P7(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 768, 1024, 512],
                 out_dim=None,
                 width=1.0,
                 depth=1.0,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(ELAN_PaFPN_P7, self).__init__()
        print('==============================')
        print('FPN: {}'.format("ELAN_PaFPN-P7"))

        self.in_dims = in_dims
        self.width = width
        self.depth = depth
        c3, c4, c5, c6, c7 = in_dims

        # top dwon
        ## P7 -> P6
        self.cv0 = Conv(c7, int(512 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv1 = Conv(c6, int(512 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_0 = ELANBlock(in_dim=int(512 * width) + int(512 * width),
                                     out_dim=int(512 * width),
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        ## P6 -> P5
        self.cv2 = Conv(int(512 * width), int(384 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv3 = Conv(c5, int(384 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_1 = ELANBlock(in_dim=int(384 * width) + int(384 * width),
                                     out_dim=int(384 * width),
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        ## P5 -> P4
        self.cv4 = Conv(int(384 * width), int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv5 = Conv(c4, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_2 = ELANBlock(in_dim=int(256 * width) + int(256 * width),
                                     out_dim=int(256 * width),
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P3
        self.cv6 = Conv(int(256 * width), int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv7 = Conv(c3, int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_3 = ELANBlock(in_dim=int(128 * width) + int(128 * width),
                                     out_dim=int(128 * width),  # 128
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # bottom up
        # P3 -> P4
        self.mp1 = DownSample(int(128 * width), out_dim=int(256 * width), 
                              act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_4 = ELANBlock(in_dim=int(256 * width) + int(256 * width),
                                     out_dim=int(256 * width),  # 256
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P5
        self.mp2 = DownSample(int(256 * width), out_dim=int(384 * width),
                              act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_5 = ELANBlock(in_dim=int(384 * width) + int(384 * width),
                                     out_dim=int(384 * width),  # 384
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)
        # P5 -> P6
        self.mp3 = DownSample(int(384 * width), out_dim=int(512 * width),
                              act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_6 = ELANBlock(in_dim=int(512 * width) + int(512 * width),
                                     out_dim=int(512 * width),  # 512
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P6 -> P7
        self.mp4 = DownSample(int(512 * width), out_dim=int(512 * width),
                              act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_7 = ELANBlock(in_dim=int(512 * width) + c7,
                                     out_dim=int(768 * width),  # 512
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)
        if out_dim is not None:
            # output proj layers
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1,
                        norm_type=norm_type, act_type=act_type)
                        for in_dim in [int(128 * width), int(256 * width), int(384 * width), int(512 * width), int(768 * width)]
                        ])
            self.out_dim = [out_dim] * 4

        else:
            self.out_layers = None
            self.out_dim = [int(128 * width), int(256 * width), int(384 * width), int(512 * width), int(768 * width)]


    def forward(self, features):
        c3, c4, c5, c6, c7 = features

        # Top down
        ## P7 -> P6
        c8 = self.cv0(c7)
        c9 = F.interpolate(c8, scale_factor=2.0)
        c10 = torch.cat([c9, self.cv1(c6)], dim=1)
        c11 = self.head_elan_0(c10)
        ## P6 -> P5
        c12 = self.cv2(c11)
        c13 = F.interpolate(c12, scale_factor=2.0)
        c14 = torch.cat([c13, self.cv3(c5)], dim=1)
        c15 = self.head_elan_1(c14)
        ## P5 -> P4
        c16 = self.cv4(c15)
        c17 = F.interpolate(c16, scale_factor=2.0)
        c18 = torch.cat([c17, self.cv5(c4)], dim=1)
        c19 = self.head_elan_2(c18)
        ## P4 -> P3
        c20 = self.cv6(c19)
        c21 = F.interpolate(c20, scale_factor=2.0)
        c22 = torch.cat([c21, self.cv7(c3)], dim=1)
        c23 = self.head_elan_3(c22)

        # Bottom up
        # p3 -> P4
        c24 = self.mp1(c23)
        c25 = torch.cat([c24, c19], dim=1)
        c26 = self.head_elan_4(c25)
        # P4 -> P5
        c27 = self.mp2(c26)
        c28 = torch.cat([c27, c15], dim=1)
        c29 = self.head_elan_5(c28)
        # P5 -> P6
        c30 = self.mp3(c29)
        c31 = torch.cat([c30, c11], dim=1)
        c32 = self.head_elan_6(c31)
        # P6 -> P7
        c33 = self.mp4(c32)
        c34 = torch.cat([c33, c7], dim=1)
        c35 = self.head_elan_7(c34)

        out_feats = [c23, c26, c29, c32, c35] # [P3, P4, P5, P6, P7]
        
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
    if model == 'elan_pafpn':
        fpn_net = ELAN_PaFPN(in_dims=in_dims,
                             out_dim=out_dim,
                             width=cfg['width'],
                             depth=cfg['depth'],
                             act_type=cfg['fpn_act'],
                             norm_type=cfg['fpn_norm'],
                             depthwise=cfg['fpn_depthwise']
                             )
    elif model == 'elan_pafpn_p6':
        fpn_net = ELAN_PaFPN_P6(in_dims=in_dims,
                                out_dim=out_dim,
                                width=cfg['width'],
                                depth=cfg['depth'],
                                act_type=cfg['fpn_act'],
                                norm_type=cfg['fpn_norm'],
                                depthwise=cfg['fpn_depthwise']
                                )
    elif model == 'elan_pafpn_p7':
        fpn_net = ELAN_PaFPN_P7(in_dims=in_dims,
                                out_dim=out_dim,
                                width=cfg['width'],
                                depth=cfg['depth'],
                                act_type=cfg['fpn_act'],
                                norm_type=cfg['fpn_norm'],
                                depthwise=cfg['fpn_depthwise']
                                )

    return fpn_net