from .elan_cspnet import build_elan_cspnet


def build_backbone(cfg):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    if cfg['backbone'] in ['elan_cspnet_nano',  'elan_cspnet_tiny',
                           'elan_cspnet_small', 'elan_cspnet_medium',
                           'elan_cspnet_large', 'elan_cspnet_huge']:
        model, feat_dim = build_elan_cspnet(cfg=cfg)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
