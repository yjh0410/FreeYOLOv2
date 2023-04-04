import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .yolo_free_v2_basic import Conv, ELANBlock
except:
    from yolo_free_v2_basic import Conv, ELANBlock


# PaFPN-ELAN
class ELAN_PaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[512, 1024, 1024],
                 out_dim=None,
                 width=1.0,
                 depth=1.0,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(ELAN_PaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("ELAN-PaFPN"))

        self.in_dims = in_dims
        self.width = width
        self.depth = depth
        c3, c4, c5 = in_dims

        # top dwon
        ## P5 -> P4
        self.reduce_layer_1 = Conv(c5, int(512 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.reduce_layer_2 = Conv(c4, int(512 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.elan_layer_1 = ELANBlock(in_dim=int(512 * width) + int(512 * width),
                                     out_dim=int(512 * width),
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P3
        self.reduce_layer_3 = Conv(int(512 * width), int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.reduce_layer_4 = Conv(c3, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.elan_layer_2 = ELANBlock(in_dim=int(256 * width) + int(256 * width),
                                     out_dim=int(256 * width),  # 128
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # bottom up
        # P3 -> P4
        self.down_layer_1 = Conv(int(256 * width), int(256 * width), k=3, p=1, s=2,
                                 act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.elan_layer_3 = ELANBlock(in_dim=int(256 * width) + int(256 * width),
                                     out_dim=int(512 * width),  # 256
                                     expand_ratio=[0.5, 0.5],
                                     depth=depth,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P5
        self.down_layer_2 = Conv(int(512 * width), int(512 * width), k=3, p=1, s=2,
                                 act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.elan_layer_4 = ELANBlock(in_dim=int(512 * width) + int(512 * width),
                                     out_dim=int(1024 * width),  # 512
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
                        for in_dim in [int(256 * width), int(512 * width), int(1024 * width)]
                        ])
            self.out_dim = [out_dim] * 3

        else:
            self.out_layers = None
            self.out_dim = [int(256 * width), int(512 * width), int(1024 * width)]


    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = self.reduce_layer_1(c5)
        c7 = self.reduce_layer_2(c4)
        c8 = torch.cat([F.interpolate(c6, scale_factor=2.0), c7], dim=1)
        c9 = self.elan_layer_1(c8)
        ## P4 -> P3
        c10 = self.reduce_layer_3(c9)
        c11 = self.reduce_layer_4(c3)
        c12 = torch.cat([F.interpolate(c10, scale_factor=2.0), c11], dim=1)
        c13 = self.elan_layer_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.down_layer_1(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.elan_layer_3(c15)
        # P4 -> P5
        c17 = self.down_layer_2(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.elan_layer_4(c18)

        out_feats = [c13, c16, c19] # [P3, P4, P5]
        
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
    return fpn_net


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'fpn': 'elan_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
    }
    model = build_fpn(cfg, in_dims=[512, 1024, 1024])
    pyramid_feats = [torch.randn(1, 512, 80, 80), torch.randn(1, 1024, 40, 40), torch.randn(1, 1024, 20, 20)]
    t0 = time.time()
    outputs = model(pyramid_feats)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    print('==============================')
    flops, params = profile(model, inputs=(pyramid_feats, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))