from .elannet import build_elannet
from .darknet import build_darknet53


def build_backbone(cfg):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    # build ELANNet
    if cfg['backbone'] == 'elannet':
        model, feat_dim = build_elannet(cfg=cfg)
    
    # build (CSP-)DarkNet
    elif cfg['backbone'] == 'darknet53':
        model, feat_dim = build_darknet53(cfg=cfg)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
