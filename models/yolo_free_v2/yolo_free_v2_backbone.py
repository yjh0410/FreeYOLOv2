import torch
import torch.nn as nn

try:
    from .yolo_free_v2_basic import Conv, ELAN_CSP_Block
except:
    from yolo_free_v2_basic import Conv, ELAN_CSP_Block


# ---------------------------- ImageNet pretrained weights ----------------------------
model_urls = {
    'elan_cspnet_nano': "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_nano.pth",
    'elan_cspnet_small': None,
    'elan_cspnet_medium': None,
    'elan_cspnet_large': "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_large.pth",
    'elan_cspnet_huge': None,
}


# ---------------------------- Basic functions ----------------------------
## ELAN-CSPNet
class ELAN_CSPNet(nn.Module):
    def __init__(self, width=1.0, depth=1.0, ratio=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELAN_CSPNet, self).__init__()
        self.feat_dims = [int(256 * width), int(512 * width), int(512 * width * ratio)]
        
        # stride = 2
        self.layer_1 =  Conv(3, int(64*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type)
        
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv(int(64*width), int(128*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(128*width), int(128*width), nblocks=int(3*depth), shortcut=True,
                           act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv(int(128*width), int(256*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(256*width), int(256*width), nblocks=int(6*depth), shortcut=True,
                           act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv(int(256*width), int(512*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(512*width), int(512*width), nblocks=int(6*depth), shortcut=True,
                           act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv(int(512*width), int(512*width*ratio), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(512*width*ratio), int(512*width*ratio), nblocks=int(3*depth), shortcut=True,
                           act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


# ---------------------------- Functions ----------------------------
## load pretrained weight
def load_weight(model, model_name):
    # load weight
    print('Loading pretrained weight ...')
    url = model_urls[model_name]
    if url is not None:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)
    else:
        print('No pretrained for {}'.format(model_name))

    return model


## build ELAN-Net
def build_backbone(cfg, pretrained=False): 
    # model
    backbone = ELAN_CSPNet(
        width=cfg['width'],
        depth=cfg['depth'],
        ratio=cfg['ratio'],
        act_type=cfg['bk_act'],
        norm_type=cfg['bk_norm'],
        depthwise=cfg['bk_dpw']
        )
    feat_dims = backbone.feat_dims
        
    # check whether to load imagenet pretrained weight
    if pretrained:
        if cfg['width'] == 0.25 and cfg['depth'] == 0.34 and cfg['ratio'] == 2.0:
            backbone = load_weight(backbone, model_name='elan_cspnet_nano')
        elif cfg['width'] == 0.5 and cfg['depth'] == 0.34 and cfg['ratio'] == 2.0:
            backbone = load_weight(backbone, model_name='elan_cspnet_small')
        elif cfg['width'] == 0.75 and cfg['depth'] == 0.67 and cfg['ratio'] == 1.5:
            backbone = load_weight(backbone, model_name='elan_cspnet_medium')
        elif cfg['width'] == 1.0 and cfg['depth'] == 1.0 and cfg['ratio'] == 1.0:
            backbone = load_weight(backbone, model_name='elan_cspnet_large')
        elif cfg['width'] == 1.25 and cfg['depth'] == 1.34 and cfg['ratio'] == 1.0:
            backbone = load_weight(backbone, model_name='elan_cspnet_huge')

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
    }
    model, feats = build_backbone(cfg)
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    x = torch.randn(1, 3, 640, 640)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))