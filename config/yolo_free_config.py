# yolo-free config


yolo_free_config = {
    'yolo_free_nano': {
        # input
        'train_size': 800,
        'test_size': 640,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640, 672,
                        704, 736, 768, 800],
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
        'backbone': 'csp_elannet',
        'width': 0.25,
        'depth': 0.33,
        'bk_act': 'lrelu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'csp_elan_pafpn',
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
        'base_lr': 0.01 / 64.,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
        },

    'yolo_free_small': {
        # input
        'train_size': 800,
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
        'backbone': 'csp_elannet',
        'width': 0.50,
        'depth': 0.34,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'csp_elan_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': True,
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
        'base_lr': 0.01 / 64.,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
        },

    'yolo_free_medium': {
        # input
        'train_size': 800,
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
        'backbone': 'csp_elannet',
        'width': 0.75,
        'depth': 0.67,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'csp_elan_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': True,
        # head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        'reg_max': 0,
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
        'base_lr': 0.01 / 64.,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
        },

    'yolo_free_large': {
        # input
        'train_size': 800,
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
        'backbone': 'csp_elannet',
        'width': 1.0,
        'depth': 1.0,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'csp_elan_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': True,
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
        'base_lr': 0.01 / 64.,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
        },

    'yolo_free_huge': {
        # input
        'train_size': 800,
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
        'backbone': 'csp_elannet',
        'width': 1.25,
        'depth': 1.34,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_depthwise': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'csp_elan_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': True,
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
        'base_lr': 0.01 / 64.,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
        },

}