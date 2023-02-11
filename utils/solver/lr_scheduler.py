import math
import torch


def build_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    if cfg['scheduler'] == 'cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg['lrf'] - 1) + 1
    elif cfg['scheduler'] == 'sonstant':
        lf = lambda x: 1.0
    else:
        print('unknown lr scheduler, use Cosine defaulted')
        exit(0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    return scheduler, lf
