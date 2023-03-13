# yolov4-e config


yolov4_ex_config = {
    'yolov4_ex': {
        # input
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640, 672,
                        704, 736, 768, 800,
                        832, 864, 896],
        'mosaic_prob': 1.0,
        'mixup_prob': 0.15,
        'format': 'RGB',
        'trans_config': {'degrees': 0.0,
                          'translate': 0.2,
                          'scale': 0.9,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4
                          },
        # model
        'backbone': 'darknet53',
        'pretrained': False,
        'csp_block': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'width': 1.0,
        'depth': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'csp_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'spp_block': True,
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'no_aug_epoch': 10,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # lr schedule
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'scheduler': 'cosine',
        'lr0': 0.01,
        'lrf': 0.01,
        },

}