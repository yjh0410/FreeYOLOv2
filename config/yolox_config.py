# yolo-free config


yolox_config = {
    # P5
    'yolox_nano': {
        # input
        'mosaic_prob': 1.0,
        'mixup_prob': 0.5,
        'format': 'RGB',
        'trans_config': {'degrees': 0.0,
                          'translate': 0.2,
                          'scale': 0.9,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          'mosaic_type': 'yolov5_mosaic',
                          'mixup_type': 'yolox_mixup',
                          'mixup_scale': [0.5, 1.5]
                          },
        # backbone
        'backbone': 'cspdarknet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 0.25,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        # fpn
        'fpn': 'csp_pafpn',
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
        # matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # loss weight
        'cls_loss': 'bce',
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'clip_grad': 10,
        # model EMA
        'ema_decay': 0.9999,
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,
        'lrf': 0.01,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolox_small': {
        # input
        'mosaic_prob': 1.0,
        'mixup_prob': 1.0,
        'format': 'RGB',
        'trans_config': {'degrees': 0.0,
                          'translate': 0.2,
                          'scale': 0.9,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          'mosaic_type': 'yolov5_mosaic',
                          'mixup_type': 'yolox_mixup',
                          'mixup_scale': [0.5, 1.5]
                          },
        # backbone
        'backbone': 'cspdarknet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 0.50,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        # fpn
        'fpn': 'csp_pafpn',
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
        # matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # loss weight
        'cls_loss': 'bce',
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'clip_grad': 10,
        # model EMA
        'ema_decay': 0.9999,
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,
        'lrf': 0.01,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolox_medium': {
        # input
        'mosaic_prob': 1.0,
        'mixup_prob': 1.0,
        'format': 'RGB',
        'trans_config': {'degrees': 0.0,
                          'translate': 0.2,
                          'scale': 0.9,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          'mosaic_type': 'yolov5_mosaic',
                          'mixup_type': 'yolox_mixup',
                          'mixup_scale': [0.5, 1.5]
                          },
        # model
        'backbone': 'cspdarknet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 0.75,
        'depth': 0.67,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'csp_pafpn',
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
        # matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # loss weight
        'cls_loss': 'bce',
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'clip_grad': 10,
        # model EMA
        'ema_decay': 0.9999,
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,
        'lrf': 0.01,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolox_large': {
        # input
        'mosaic_prob': 1.0,
        'mixup_prob': 1.0,
        'format': 'RGB',
        'trans_config': {'degrees': 0.0,
                          'translate': 0.2,
                          'scale': 0.9,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          'mosaic_type': 'yolov5_mosaic',
                          'mixup_type': 'yolox_mixup',
                          'mixup_scale': [0.5, 1.5]
                          },
        # model
        'backbone': 'cspdarknet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 1.0,
        'depth': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'csp_pafpn',
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
        # matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # loss weight
        'cls_loss': 'bce',
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'clip_grad': 10,
        # model EMA
        'ema_decay': 0.9999,
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,
        'lrf': 0.01,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolox_huge': {
        # input
        'mosaic_prob': 1.0,
        'mixup_prob': 1.0,
        'format': 'RGB',
        'trans_config': {'degrees': 0.0,
                          'translate': 0.2,
                          'scale': 0.9,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          'mosaic_type': 'yolov5_mosaic',
                          'mixup_type': 'yolox_mixup',
                          'mixup_scale': [0.5, 1.5]
                          },
        # model
        'backbone': 'cspdarknet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'p6_feat': False,
        'p7_feat': False,
        'width': 1.25,
        'depth': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'csp_pafpn',
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
        # matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # loss weight
        'cls_loss': 'bce',
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'clip_grad': 10,
        # model EMA
        'ema_decay': 0.9999,
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,
        'lrf': 0.01,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

}