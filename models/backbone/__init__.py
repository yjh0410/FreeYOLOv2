from .elannet import build_elannet
from .darknet import build_darknet53


def build_backbone(cfg, trainable=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    # imagenet pretrained
    pretrained = cfg['pretrained'] and trainable

    if cfg['backbone'] in ['elannet_nano',  'elannet_tiny',
                           'elannet_small', 'elannet_medium',
                           'elannet_large', 'elannet_huge']:
        model, feat_dim = build_elannet(
            cfg=cfg,
            pretrained=pretrained
        )

    elif cfg['backbone'] == 'darknet53':
        model, feat_dim = build_darknet53(
            cfg=cfg,
            pretrained=pretrained
        )

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
