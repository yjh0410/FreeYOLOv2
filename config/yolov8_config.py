# yolo-free config


yolov8_config = {
    # P5
    'yolov8_nano': {
        # input
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
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'lrelu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'elan_csp_pafpn',
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
        'cls_loss': 'bce',         # optional: bce, vfl
        'loss_cls_weight': 0.5,
        'loss_iou_weight': 7.5,
        'loss_dfl_weight': 1.5,
        # training configuration
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',      # optional: sgd, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        # model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,              # SGD: 0.01;     AdamW: 0.004
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolov8_small': {
        # input
        'mosaic_prob': 1.0,
        'mixup_prob': 0.0,
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
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 0.50,
        'depth': 0.34,
        'ratio': 2.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'elan_csp_pafpn',
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
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',      # optional: sgd, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        # model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,              # SGD: 0.01;     AdamW: 0.004
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolov8_medium': {
        # input
        'mosaic_prob': 1.0,
        'mixup_prob': 0.1,
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
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 0.75,
        'depth': 0.67,
        'ratio': 1.5,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'elan_csp_pafpn',
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
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',      # optional: sgd, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        # model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,              # SGD: 0.01;     AdamW: 0.004
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolov8_large': {
        # input
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
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'elan_csp_pafpn',
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
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',      # optional: sgd, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        # model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,              # SGD: 0.01;     AdamW: 0.004
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolov8_huge': {
        # input
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
        'backbone': 'elannet',
        'pretrained': False,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 1.25,
        'depth': 1.00,
        'ratio': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'elan_csp_pafpn',
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
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',      # optional: sgd, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        # model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,              # SGD: 0.01;     AdamW: 0.004
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.05
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

}