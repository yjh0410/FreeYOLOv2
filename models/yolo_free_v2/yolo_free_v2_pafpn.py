import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .yolo_free_v2_basic import Conv, ELAN_CSP_Block
except:
    from yolo_free_v2_basic import Conv, ELAN_CSP_Block


# PaFPN-ELAN
class ELAN_CSP_PaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 512],
                 width=1.0,
                 depth=1.0,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(ELAN_CSP_PaFPN, self).__init__()
        print('==============================')
        print('FPN: {}'.format("ELAN_PaFPN"))
        self.in_dims = in_dims
        self.width = width
        self.depth = depth
        c3, c4, c5 = in_dims

        # top dwon
        ## P5 -> P4
        self.cv1 = Conv(c5, int(256*width), k=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_1 = ELAN_CSP_Block(in_dim=int(256*width) + c4,
                                          out_dim=int(256*width),
                                          expand_ratio=0.5,
                                          nblocks=int(3*depth),
                                          shortcut=False,
                                          depthwise=depthwise,
                                          norm_type=norm_type,
                                          act_type=act_type
                                          )

        # P4 -> P3
        self.cv2 = Conv(int(256*width), int(128*width), k=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_2 = ELAN_CSP_Block(in_dim=int(128*width) + c3,
                                          out_dim=int(128*width),
                                          expand_ratio=0.5,
                                          nblocks=int(3*depth),
                                          shortcut=False,
                                          depthwise=depthwise,
                                          norm_type=norm_type,
                                          act_type=act_type
                                          )

        # bottom up
        # P3 -> P4
        self.mp1 = Conv(int(128*width), int(128*width), k=3, p=1, s=2,
                        act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_3 = ELAN_CSP_Block(in_dim=int(128*width) + int(128*width),
                                          out_dim=int(256*width),
                                          expand_ratio=0.5,
                                          nblocks=int(3*depth),
                                          shortcut=False,
                                          depthwise=depthwise,
                                          norm_type=norm_type,
                                          act_type=act_type
                                          )

        # P4 -> P5
        self.mp2 = Conv(int(256 * width), int(256 * width), k=3, p=1, s=2,
                        act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_4 = ELAN_CSP_Block(in_dim=int(256 * width) + int(256 * width),
                                          out_dim=int(512 * width),
                                          expand_ratio=0.5,
                                          nblocks=int(3*depth),
                                          shortcut=False,
                                          depthwise=depthwise,
                                          norm_type=norm_type,
                                          act_type=act_type
                                          )

        self.out_dim = [int(128 * width), int(256 * width), int(512 * width)]


    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_elan_1(c8)
        ## P4 -> P3
        c10 = self.cv2(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_elan_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.mp1(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_elan_3(c15)
        # P4 -> P5
        c17 = self.mp2(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_elan_4(c18)

        out_feats = [c13, c16, c19] # [P3, P4, P5]
        
        return out_feats


def build_fpn(cfg, in_dims):
    model = cfg['fpn']
    # build neck
    if model == 'elan_csp_pafpn':
        fpn_net = ELAN_CSP_PaFPN(in_dims=in_dims,
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
        'fpn': 'elan_csp_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
    }
    model = build_fpn(cfg, in_dims=[256, 512, 512])
    pyramid_feats = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 512, 20, 20)]
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