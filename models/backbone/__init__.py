from .csp_elannet import build_csp_elannet


def build_backbone(cfg):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    if cfg['backbone'] == 'csp_elannet':
        model, feat_dim = build_csp_elannet(cfg)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
