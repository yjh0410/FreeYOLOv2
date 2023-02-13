# yolo-free config


yolo_free_v2_config = {
    'yolo_free_v2_nano': {
        # input
        'train_size': 640,
        'test_size': 416,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640],
        'mosaic_prob': 1.0,
        'mixup_prob': 0.0,
        'format': 'RGB',
        'trans_config': {'degrees': 0.0,
                          'translate': 0.1,
                          'scale': 0.5,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4
                          },
        # model
        'backbone': 'elannet_nano',
        'bk_act': 'lrelu',
        'bk_norm': 'BN',
        'bk_dpw': True,
        'width': 0.25,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': True,
        # fpn
        'fpn': 'elan_pafpn',
        'fpn_act': 'lrelu',
        'fpn_norm': 'BN',
        'fpn_depthwise': True,
        # head
        'head': 'decoupled_head',
        'head_act': 'lrelu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': True,
        'reg_max': 16,
        # matcher
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        'loss_dfl_weight': 1.5,
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

    'yolo_free_v2_tiny': {
        # input
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640, 672,
                        704, 736, 768, 800],
        'mosaic_prob': 1.0,
        'mixup_prob': 0.05,
        'format': 'RGB',
        'trans_config': {'degrees': 0.0,
                          'translate': 0.1,
                          'scale': 0.5,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4
                          },
        # model
        'backbone': 'elannet_tiny',
        'bk_act': 'lrelu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.375,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'elan_pafpn',
        'fpn_act': 'lrelu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_act': 'lrelu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        'reg_max': 16,
        # matcher
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        'loss_dfl_weight': 1.5,
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

    'yolo_free_v2_small': {
        # input
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640, 672,
                        704, 736, 768, 800],
        'mosaic_prob': 1.0,
        'mixup_prob': 0.05,
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
        'backbone': 'elannet_small',
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.50,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'elan_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        'reg_max': 16,
        # matcher
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        'loss_dfl_weight': 1.5,
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

    'yolo_free_v2_medium': {
        # input
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640, 672,
                        704, 736, 768, 800],
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
        'backbone': 'elannet_medium',
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.75,
        'depth': 0.67,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'elan_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        'reg_max': 16,
        # matcher
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        'loss_dfl_weight': 1.5,
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

    'yolo_free_v2_large': {
        # input
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640, 672,
                        704, 736, 768, 800],
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
        'backbone': 'elannet_large',
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
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
        'fpn': 'elan_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        'reg_max': 16,
        # matcher
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        'loss_dfl_weight': 1.5,
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

    'yolo_free_v2_huge': {
        # input
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640, 672,
                        704, 736, 768, 800],
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
        'backbone': 'elannet_huge',
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.25,
        'depth': 1.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'elan_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        'reg_max': 16,
        # matcher
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        'loss_dfl_weight': 1.5,
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