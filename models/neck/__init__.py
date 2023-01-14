from .spp import SPPFBlockCSP
from .elan_pafpn import ELAN_PaFPN


def build_fpn(cfg, in_dims):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'elan_pafpn':
        fpn_net = ELAN_PaFPN(in_dims=in_dims,
                             width=cfg['width'],
                             depth=cfg['depth'],
                             act_type=cfg['fpn_act'],
                             norm_type=cfg['fpn_norm'],
                             depthwise=cfg['fpn_depthwise']
                             )

    return fpn_net


def build_neck(cfg, in_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'sppf_block_csp':
        neck = SPPFBlockCSP(
            in_dim=in_dim,
            width=cfg['width'],
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm'],
            depthwise=cfg['neck_depthwise']
            )

    return neck
    