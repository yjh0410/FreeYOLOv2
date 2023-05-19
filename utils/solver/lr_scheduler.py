import math
import torch


def build_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    print('==============================')
    print('Lr Scheduler: {}'.format(cfg['scheduler']))

    if cfg['scheduler'] == 'cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg['lrf'] - 1) + 1
        
    elif cfg['scheduler'] == 'linear':
        lf = lambda x: (1 - x / epochs) * (1.0 - cfg['lrf']) + cfg['lrf']

    elif cfg['scheduler'] == 'cos_linear':
        lf = lambda x: (1 - x / epochs) * (1.0 - cfg['lrf']) + cfg['lrf'] if x > epochs // 2 else ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg['lrf'] - 1) + 1
        
    else:
        print('unknown lr scheduler.')
        exit(0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    return scheduler, lf
