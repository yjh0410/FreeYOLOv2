from .elannet import build_elannet


def build_backbone(cfg, trainable=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    # imagenet pretrained
    pretrained = cfg['pretrained'] and trainable

    if cfg['backbone'] in ['elannet_large', 'elannet_huge', \
                           'elannet_tiny', 'elannet_nano']:
        model, feat_dim = build_elannet(
            cfg=cfg,
            pretrained=pretrained
        )

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
