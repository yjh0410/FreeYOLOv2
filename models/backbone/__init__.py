from .csp_elannet import build_csp_elannet, build_elannet


def build_backbone(cfg):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    if cfg['backbone'] == 'csp_elannet':
        model, feat_dim = build_csp_elannet(cfg)
    elif cfg['backbone'] == 'elannet':
        model, feat_dim = build_elannet(model_name='elannet', pretrained=True)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
