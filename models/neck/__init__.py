from .spp import SPPF, SPPFBlockCSP
from .elan_pafpn import ELAN_PaFPN, ELAN_PaFPN_P6


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
    if model == 'elan_pafpn_p6':
        fpn_net = ELAN_PaFPN_P6(in_dims=in_dims,
                                out_dim=out_dim,
                                width=cfg['width'],
                                depth=cfg['depth'],
                                act_type=cfg['fpn_act'],
                                norm_type=cfg['fpn_norm'],
                                depthwise=cfg['fpn_depthwise']
                                )

    return fpn_net


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'sppf':
        neck = SPPF(
            in_dim=in_dim,
            out_dim=out_dim,
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm']
            )
    elif model == 'sppf_block_csp':
        neck = SPPFBlockCSP(
            in_dim=in_dim,
            out_dim=out_dim,
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm'],
            depthwise=cfg['neck_depthwise']
            )

    return neck
    