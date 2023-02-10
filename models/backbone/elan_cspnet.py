import torch
import torch.nn as nn


model_urls = {
    "elan_cspnet_nano": None,
    "elan_cspnet_tiny": None,
    "elan_cspnet_small": None,
    "elan_cspnet_medium": None,
    "elan_cspnet_large": None,
    "elan_cspnet_huge": None,
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


# BottleNeck
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)  # hidden channels            
        self.cv1 = Conv(in_dim, inter_dim, k=3, p=1, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.cv2 = Conv(inter_dim, out_dim, k=3, p=1, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h


# ELAN-CSP-Block
class ELAN_CSP_Block(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 nblocks=1,
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(ELAN_CSP_Block, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.m = nn.Sequential(*(
            Bottleneck(inter_dim, inter_dim, 1.0, shortcut, depthwise, act_type, norm_type)
            for _ in range(nblocks)))
        self.cv3 = Conv((2 + nblocks) * inter_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        out = list([x1, x2])

        out.extend(m(out[-1]) for m in self.m)

        out = self.cv3(torch.cat(out, dim=1))

        return out


# DownSample
class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        inter_dim = out_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


# ELAN-CSPNet
class ELAN_CSPNet(nn.Module):
    def __init__(self, width=1.0, depth=1.0, ratio=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELAN_CSPNet, self).__init__()
        self.feat_dims = [int(256*width), int(512*width), int(512*width*ratio)]

        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv(3, int(64*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Conv(int(64*width), int(64*width), k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise) # P1/2
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv(int(64*width), int(128*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(128*width), int(128*width), expand_ratio=0.5, nblocks=int(3*depth),
                           shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=int(128*width), out_dim=int(256*width), act_type=act_type, norm_type=norm_type),             
            ELAN_CSP_Block(int(256*width), int(256*width), expand_ratio=0.5, nblocks=int(6*depth),
                           shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=int(256*width), out_dim=int(512*width), act_type=act_type, norm_type=norm_type),             
            ELAN_CSP_Block(int(512*width), int(512*width), expand_ratio=0.5, nblocks=int(6*depth),
                           shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=int(512*width), out_dim=int(512*width*ratio), act_type=act_type, norm_type=norm_type),             
            ELAN_CSP_Block(int(512*width*ratio), int(512*width*ratio), expand_ratio=0.5, nblocks=int(3*depth),
                           shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
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
def build_elan_cspnet(cfg, pretrained=False): 
    # model
    backbone = ELAN_CSPNet(width=cfg['width'], depth=cfg['depth'], ratio=cfg['ratio'],
                           act_type=cfg['bk_act'], norm_type=cfg['bk_norm'], depthwise=cfg['bk_dpw'])
    feat_dims = backbone.feat_dims

    # load weight
    if pretrained:
        url = model_urls[cfg['backbone']]
        if url is not None:
            print('Loading pretrained weight ...')
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = backbone.state_dict()
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

            backbone.load_state_dict(checkpoint_state_dict)
        else:
            print('No backbone pretrained: {}'.format(cfg['backbone']))

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    from thop import profile
    cfg = {
        'backbone': 'elan_cspnet_tiny',
        'bk_act': 'lrelu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0
    }
    model, feats = build_elan_cspnet(cfg, pretrained=True)
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
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
