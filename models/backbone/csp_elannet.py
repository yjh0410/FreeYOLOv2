import torch
import torch.nn as nn


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


# Bottleneck
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 act_type='silu',
                 norm_type='BN',
                 shortcut=False,
                 depthwise=False):
        super().__init__()
        self.cv1 = Conv(in_dim, in_dim, k=3, p=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, in_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.shortcut = shortcut

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h


# ELANBlock - same to 'C2f' proposed by YOLOv8
class CSP_ELANBlock(nn.Module): 
    def __init__(self,
                 in_dim,
                 out_dim,
                 depth=1,
                 act_type='silu',
                 norm_type='BN',
                 shortcut=False,
                 depthwise=False):
        super(CSP_ELANBlock, self).__init__()
        inter_dim = in_dim // 2
        self.cv1 = Conv(in_dim, 2*inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(*[
            Bottleneck(inter_dim, act_type=act_type, norm_type=norm_type,
                       shortcut=shortcut, depthwise=depthwise)
            for _ in range(depth)
            ])
        self.cv3 = nn.Sequential(*[
            Bottleneck(inter_dim, act_type=act_type, norm_type=norm_type,
                       shortcut=shortcut, depthwise=depthwise)
            for _ in range(depth)
            ])
        self.cv4 = Conv(inter_dim*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1, x2 = self.cv1(x).chunk(2, 1)
        x3 = self.cv2(x2)
        x4 = self.cv3(x3)
        y = torch.cat([x1, x2, x3, x4], dim=1)

        return self.cv4(y)


# DownSample
class DownSample(nn.Module):
    def __init__(self, in_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # maxpooling branch
        inter_dim = in_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        # conv-s2 branch
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
        )


    def forward(self, x):
        # [B, C, H, W] -> [B, C, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, 2C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


# CSP_ELANNet - inspired by YOLOv7&v8
class CSP_ELANNet(nn.Module):
    def __init__(self, width=1.0, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(CSP_ELANNet, self).__init__()
        self.feats_dim = [int(256*width), int(512*width), int(1024*width)]

        # --------- Network Parameters ----------
        self.layer_1 = nn.Sequential(
            Conv(3, int(64*width), k=3, p=1, s=2,
                 act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            Conv(int(64*width), int(64*width), k=3, p=1,
                 act_type=act_type, norm_type=norm_type, depthwise=depthwise)           # P1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(int(64*width), int(64*width), k=3, p=1, s=2,
                 act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            CSP_ELANBlock(in_dim=int(64*width), out_dim=int(128*width), depth=int(3*depth), shortcut=True,
                          act_type=act_type, norm_type=norm_type, depthwise=depthwise)  # P2/4
        )
        self.layer_3 = nn.Sequential(   
            DownSample(in_dim=int(128*width), act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            CSP_ELANBlock(in_dim=int(128*width), out_dim=int(256*width), depth=int(6*depth), shortcut=True,
                          act_type=act_type, norm_type=norm_type, depthwise=depthwise)  # P3/8
        )
        self.layer_4 = nn.Sequential(   
            DownSample(in_dim=int(256*width), act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            CSP_ELANBlock(in_dim=int(256*width), out_dim=int(512*width), depth=int(9*depth), shortcut=True,
                          act_type=act_type, norm_type=norm_type, depthwise=depthwise)  # P4/16
        )
        self.layer_5 = nn.Sequential(   
            DownSample(in_dim=int(512*width), act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            CSP_ELANBlock(in_dim=int(512*width), out_dim=int(1024*width), depth=int(3*depth), shortcut=True,
                          act_type=act_type, norm_type=norm_type, depthwise=depthwise)  # P5/32
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = {
            'layer2': c3,
            'layer3': c4,
            'layer4': c5
        }
        return outputs



# build ELAN-Net
def build_csp_elannet(cfg):
    # model
    backbone = CSP_ELANNet(cfg['width'], cfg['depth'], cfg['bk_act'], cfg['bk_norm'], cfg['bk_depthwise'])
    feats_dim = backbone.feats_dim

    return backbone, feats_dim


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'width': 1.00,
        'depth': 1.00,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False
    }
    model, feats = build_csp_elannet(cfg)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for k in outputs.keys():
        print(outputs[k].shape)

    x = torch.randn(1, 3, 224, 224)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

