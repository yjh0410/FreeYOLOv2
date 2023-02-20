import torch.nn as nn
from .conv import Conv


class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 kernel=[1, 3],
                 shortcut=False,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)  # hidden channels            
        self.cv1 = Conv(in_dim, inter_dim, k=kernel[0], p=kernel[0]//2,
                        norm_type=norm_type, act_type=act_type,
                        depthwise=False if kernel[0] == 1 else depthwise)
        self.cv2 = Conv(inter_dim, out_dim, k=kernel[1], p=kernel[1]//2,
                        norm_type=norm_type, act_type=act_type,
                        depthwise=False if kernel[1] == 1 else depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h