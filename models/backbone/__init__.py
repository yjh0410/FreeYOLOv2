from .elannet import build_elannet


def build_backbone(cfg):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    if cfg['backbone'] in ['elannet_nano',  'elannet_tiny',
                           'elannet_small', 'elannet_medium',
                           'elannet_large', 'elannet_huge']:
        model, feat_dim = build_elannet(cfg=cfg)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
