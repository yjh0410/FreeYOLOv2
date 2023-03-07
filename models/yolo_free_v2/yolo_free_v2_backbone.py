import torch
import torch.nn as nn

try:
    from .yolo_free_v2_basic import ELANBlock, DownSample
except:
    from yolo_free_v2_basic import ELANBlock, DownSample


model_urls = {
    'elannet_pico': "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_nano.pth",
    'elannet_nano': "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_tiny.pth",
    'elannet_tiny': None,
    'elannet_small': "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_small.pth",
    'elannet_medium': "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_medium.pth",
    'elannet_large': "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_large.pth",
    'elannet_huge': "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_huge.pth",

}


def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)


def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)


# ---------------------------- Basic module ----------------------------
# Basic conv layer
class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='silu',      # activation
                 norm_type='BN',       # normalization
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        if depthwise:
            # depthwise conv
            convs.append(nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=p, dilation=d, groups=c1, bias=False))
            convs.append(get_norm(norm_type, c1))
            if act_type is not None:
                convs.append(get_activation(act_type))

            # pointwise conv
            convs.append(nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, dilation=d, groups=1, bias=False))
            convs.append(get_norm(norm_type, c2))
            if act_type is not None:
                convs.append(get_activation(act_type))

        else:
            convs.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, dilation=d, groups=1, bias=False))
            convs.append(get_norm(norm_type, c2))
            if act_type is not None:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


# ---------------------------- Backbones ----------------------------
# ELANNet-P5
class ELANNet(nn.Module):
    def __init__(self, width=1.0, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNet, self).__init__()
        self.feat_dims = [int(512 * width), int(1024 * width), int(1024 * width)]
        
        # P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, int(64*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Conv(int(64*width), int(64*width), k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P2/4
        self.layer_2 = nn.Sequential(   
            Conv(int(64*width), int(128*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(in_dim=int(128*width), out_dim=int(256*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=int(256*width), out_dim=int(256*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(256*width), out_dim=int(512*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=int(512*width), out_dim=int(512*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(512*width), out_dim=int(1024*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=int(1024*width), out_dim=int(1024*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(1024*width), out_dim=int(1024*width), expand_ratio=0.25, depth=depth,
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


# ELANNet-P6
class ELANNet_P6(nn.Module):
    def __init__(self, width=1.0, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNet_P6, self).__init__()
        self.p5_stage_dim = 1024
        self.feat_dims = [int(256 * width), int(512 * width), int(768 * width), int(1024 * width)]
        
        # P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, int(64*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Conv(int(64*width), int(64*width), k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P2/4
        self.layer_2 = nn.Sequential(   
            Conv(int(64*width), int(128*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(in_dim=int(128*width), out_dim=int(128*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=int(128*width), out_dim=int(256*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(256*width), out_dim=int(256*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=int(256*width), out_dim=int(512*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(512*width), out_dim=int(512*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=int(512*width), out_dim=int(768*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(768*width), out_dim=int(768*width), expand_ratio=0.5, depth=depth,
                    act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P6/64
        self.layer_6 = nn.Sequential(
            DownSample(in_dim=int(768*width), out_dim=int(1024*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(1024*width), out_dim=int(1024*width), expand_ratio=0.5, depth=depth,
                    act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        c6 = self.layer_6(c5)

        outputs = [c3, c4, c5, c6]

        return outputs


# ELANNet-P7
class ELANNet_P7(nn.Module):
    def __init__(self, width=1.0, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNet_P7, self).__init__()
        self.p5_stage_dim = 1024
        self.feat_dims = [int(256 * width), int(512 * width), int(768 * width), int(1024 * width), int(1024 * width)]
        
        # P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, int(64*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Conv(int(64*width), int(64*width), k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P2/4
        self.layer_2 = nn.Sequential(   
            Conv(int(64*width), int(128*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(in_dim=int(128*width), out_dim=int(128*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=int(128*width), out_dim=int(256*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(256*width), out_dim=int(256*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=int(256*width), out_dim=int(512*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(512*width), out_dim=int(512*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=int(512*width), out_dim=int(768*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(768*width), out_dim=int(768*width), expand_ratio=0.5, depth=depth,
                    act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P6/64
        self.layer_6 = nn.Sequential(
            DownSample(in_dim=int(768*width), out_dim=int(1024*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(1024*width), out_dim=int(1024*width), expand_ratio=0.5, depth=depth,
                    act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P7/128
        self.layer_7 = nn.Sequential(
            DownSample(in_dim=int(1024*width), out_dim=int(1024*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(1024*width), out_dim=int(1024*width), expand_ratio=0.25, depth=depth,
                    act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        c6 = self.layer_6(c5)
        c7 = self.layer_7(c6)

        outputs = [c3, c4, c5, c6, c7]

        return outputs


# ---------------------------- Functions ----------------------------
# build ELAN-Net
def build_backbone(cfg): 
    # model
    if cfg['p6_feat']:
        if not cfg['p7_feat']:
            backbone = ELANNet_P6(
                width=cfg['width'],
                depth=cfg['depth'],
                act_type=cfg['bk_act'],
                norm_type=cfg['bk_norm'],
                depthwise=cfg['bk_dpw']
                )
        else:
            backbone = ELANNet_P7(
                width=cfg['width'],
                depth=cfg['depth'],
                act_type=cfg['bk_act'],
                norm_type=cfg['bk_norm'],
                depthwise=cfg['bk_dpw']
                )
    else:
        backbone = ELANNet(
            width=cfg['width'],
            depth=cfg['depth'],
            act_type=cfg['bk_act'],
            norm_type=cfg['bk_norm'],
            depthwise=cfg['bk_dpw']
            )
        
        # check whether to load imagenet pretrained weight
        if cfg['pretrained']:
            if cfg['width'] == 0.25 and cfg['depth'] == 0.34 and cfg['bk_dpw']:
                arch = 'elannet_pico'
            elif cfg['width'] == 0.25 and cfg['depth'] == 0.34 and not cfg['bk_dpw']:
                arch = 'elannet_nano'
            elif cfg['width'] == 0.375 and cfg['depth'] == 0.34 and not cfg['bk_dpw']:
                arch = 'elannet_tiny'
            elif cfg['width'] == 0.5 and cfg['depth'] == 0.34 and not cfg['bk_dpw']:
                arch = 'elannet_small'
            elif cfg['width'] == 0.75 and cfg['depth'] == 0.67 and not cfg['bk_dpw']:
                arch = 'elannet_medium'
            elif cfg['width'] == 1.0 and cfg['depth'] == 1.0 and not cfg['bk_dpw']:
                arch = 'elannet_large'
            elif cfg['width'] == 1.25 and cfg['depth'] == 1.34 and not cfg['bk_dpw']:
                arch = 'elannet_huge'
            backbone = load_weight(backbone, model_name=arch)

    feat_dims = backbone.feat_dims

    return backbone, feat_dims


# load pretrained weight
def load_weight(model, model_name):
    # load weight
    print('Load pretrained weight {}'.format(model_name))
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


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'bk_act': 'lrelu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 0.25,
        'depth': 0.34,
        'pretrained': True
    }
    model, feats = build_backbone(cfg)
    x = torch.randn(1, 3, 256, 256)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for out in outputs:
        print(out.shape)

    x = torch.randn(1, 3, 256, 256)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))