import torch
import torch.nn as nn

try:
    from .yolo_free_v2_basic import ELAN_CSP_Block
except:
    from yolo_free_v2_basic import ELAN_CSP_Block


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
# ELAN-CSPNet
class ELAN_CSPNet(nn.Module):
    def __init__(self, width=1.0, depth=1.0, ratio=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELAN_CSPNet, self).__init__()
        self.feat_dims = [int(256 * width), int(512 * width), int(512 * width * ratio)]
        
        # stride = 2
        self.layer_1 = Conv(3, int(64*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type)
        
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv(int(64*width), int(128*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(128*width), int(128*width), expand_ratio=0.5, nblocks=int(3*depth),
                           shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv(int(128*width), int(256*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(256*width), int(256*width), expand_ratio=0.5, nblocks=int(6*depth),
                           shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv(int(256*width), int(512*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(512*width), int(512*width), expand_ratio=0.5, nblocks=int(6*depth),
                           shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv(int(512*width), int(512*width*ratio), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(512*width*ratio), int(512*width*ratio), expand_ratio=0.5, nblocks=int(3*depth),
                           shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
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
# build ELAN-Net
def build_backbone(cfg): 
    # model
    if cfg['p6_feat']:
        if not cfg['p7_feat']:
            # P6 backbone
            pass
        else:
            # P7 backbone
            pass
    else:
        # P5 backbone
        backbone = ELAN_CSPNet(
            width=cfg['width'],
            depth=cfg['depth'],
            ratio=cfg['ratio'],
            act_type=cfg['bk_act'],
            norm_type=cfg['bk_norm'],
            depthwise=cfg['bk_dpw']
            )
        
    feat_dims = backbone.feat_dims

    return backbone, feat_dims


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
        'ratio': 2.0,
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