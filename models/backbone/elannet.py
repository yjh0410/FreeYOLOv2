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


# ELANBlock
class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(int(3*depth))
        ])
        self.cv4 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(int(3*depth))
        ])

        self.out = Conv(inter_dim*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out


# DownSample
class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN'):
        super().__init__()
        inter_dim = out_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type)
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


# ELANNet
class ELANNet(nn.Module):
    def __init__(self, width=1.0, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNet, self).__init__()
        self.feat_dims = [int(512 * width), int(1024 * width), int(1024 * width)]
        
        # large backbone
        self.layer_1 = nn.Sequential(
            Conv(3, int(64*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Conv(int(64*width), int(64*width), k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise) # P1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(int(64*width), int(128*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(in_dim=int(128*width), out_dim=int(256*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)                     # P2/4
        )
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=int(256*width), out_dim=int(256*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(256*width), out_dim=int(512*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)                     # P3/8
        )
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=int(512*width), out_dim=int(512*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(512*width), out_dim=int(1024*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)                    # P4/16
        )

        self.layer_5 = nn.Sequential(
            DownSample(in_dim=int(1024*width), out_dim=int(1024*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(1024*width), out_dim=int(1024*width), expand_ratio=0.25, depth=depth,
                    act_type=act_type, norm_type=norm_type, depthwise=depthwise)                  # P5/32
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
        
        # large backbone
        self.layer_1 = nn.Sequential(
            Conv(3, int(64*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Conv(int(64*width), int(64*width), k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise) # P1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(int(64*width), int(128*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),             
            ELANBlock(in_dim=int(128*width), out_dim=int(128*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)                     # P2/4
        )
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=int(128*width), out_dim=int(256*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(256*width), out_dim=int(256*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)                     # P3/8
        )
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=int(256*width), out_dim=int(512*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(512*width), out_dim=int(512*width), expand_ratio=0.5, depth=depth,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise)                    # P4/16
        )

        self.layer_5 = nn.Sequential(
            DownSample(in_dim=int(512*width), out_dim=int(768*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(768*width), out_dim=int(768*width), expand_ratio=0.5, depth=depth,
                    act_type=act_type, norm_type=norm_type, depthwise=depthwise)                  # P5/32
        )
        self.layer_6 = nn.Sequential(
            DownSample(in_dim=int(768*width), out_dim=int(1024*width), act_type=act_type, norm_type=norm_type),             
            ELANBlock(in_dim=int(1024*width), out_dim=int(1024*width), expand_ratio=0.5, depth=depth,
                    act_type=act_type, norm_type=norm_type, depthwise=depthwise)                  # P5/64
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


# build ELAN-Net
def build_elannet(cfg): 
    # model
    if cfg['p6_feat']:
        backbone = ELANNet_P6(
            width=cfg['width'],
            depth=cfg['depth'],
            act_type=cfg['bk_act'],
            norm_type=cfg['bk_norm'],
            depthwise=cfg['bk_dpw']
            )
        feat_dims = backbone.feat_dims
    else:
        backbone = ELANNet(
            width=cfg['width'],
            depth=cfg['depth'],
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
        'p6_feat': True,
        'width': 1.0,
        'depth': 1.0,
    }
    model, feats = build_elannet(cfg)
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