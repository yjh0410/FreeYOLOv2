from .spp import SPPF, SPPFBlockCSP
from .elan_pafpn import ELAN_PaFPN, ELAN_PaFPN_P6, ELAN_PaFPN_P7
from .csp_pafpn import CSP_PaFPN
from .basid_fpn import BasicFPN


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
    elif model == 'csp_pafpn':
        fpn_net = CSP_PaFPN(in_dims=in_dims,
                             out_dim=out_dim,
                             width=cfg['width'],
                             depth=cfg['depth'],
                             act_type=cfg['fpn_act'],
                             norm_type=cfg['fpn_norm'],
                             depthwise=cfg['fpn_depthwise']
                             )
    elif model == 'basic_fpn':
        fpn_net = BasicFPN(in_dims=in_dims,
                            out_dim=out_dim,
                            width=cfg['width'],
                            depth=cfg['depth'],
                            act_type=cfg['fpn_act'],
                            norm_type=cfg['fpn_norm']
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
    