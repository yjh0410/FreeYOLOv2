from .spp import SPPFBlockCSP
from .csp_elan_pafpn import CSP_ELAN_PaFPN


def build_fpn(cfg):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'csp_elan_pafpn':
        fpn_net = CSP_ELAN_PaFPN(width=cfg['width'],
                                 depth=cfg['depth'],
                                 depthwise=cfg['fpn_depthwise'],
                                 act_type=cfg['fpn_act'],
                                 norm_type=cfg['fpn_norm'])

    return fpn_net


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'sppf_block_csp':
        neck = SPPFBlockCSP(
            in_dim, out_dim, 
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm'],
            depthwise=cfg['neck_depthwise']
            )

    return neck
    