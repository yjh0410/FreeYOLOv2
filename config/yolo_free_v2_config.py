# yolo-free config


yolo_free_v2_cfg = {
    # P5
    'yolo_free_v2_nano': {
        # input
        'multi_scale': [0.5, 1.25],
        'trans_config': {# Basic Augment
                          'degrees': 0.0,
                          'translate': 0.1,
                          'scale': 0.5,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          # Mosaic & Mixup
                          'mosaic_prob': 0.5,
                          'mixup_prob': 0.0,
                          'mosaic_type': 'yolov5_mosaic',
                          'mixup_type': 'yolov5_mixup',
                          'mixup_scale': [0.5, 1.5],
                          },
        # model
        'backbone': 'elannet_nano',
        'pretrained': True,
        'bk_act': 'lrelu',
        'bk_norm': 'BN',
        'bk_dpw': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': True,
        # fpn
        'fpn': 'yolov7_pafpn',
        'fpn_act': 'lrelu',
        'fpn_norm': 'BN',
        'fpn_depthwise': True,
        'nbranch': 2.0,        # number of branch in ELANBlockFPN
        'depth': 1.0,          # depth factor of each branch in ELANBlockFPN
        'width': 0.25,         # width factor of channel in FPN
        # head
        'head': 'decoupled_head',
        'head_act': 'lrelu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': True,
        # matcher
        'matcher': {'soft_center_radius': 2.5,
                    'topk_candicate': 10,
                    'iou_weight': 2.0},
        # loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # training configuration
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',        # optional: sgd, adam, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        # model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,               # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolo_free_v2_tiny': {
        # input
        'multi_scale': [0.5, 1.25],
        'trans_config': {# Basic Augment
                          'degrees': 0.0,
                          'translate': 0.1,
                          'scale': 0.5,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          # Mosaic & Mixup
                          'mosaic_prob': 1.0,
                          'mixup_prob': 0.05,
                          'mosaic_type': 'yolov5_mosaic',
                          'mixup_type': 'yolov5_mixup',
                          'mixup_scale': [0.5, 1.5],
                          },
        # model
        'backbone': 'elannet_tiny',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'yolov7_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'nbranch': 2.0,       # number of branch in ELANBlockFPN
        'depth': 1.0,         # depth factor of each branch in ELANBlockFPN
        'width': 0.5,         # width factor of channel in FPN
        # head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # matcher
        'matcher': {'soft_center_radius': 2.5,
                    'topk_candicate': 10,
                    'iou_weight': 2.0},
        # loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # training configuration
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',        # optional: sgd, adam, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        # model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,               # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolo_free_v2_large': {
        # input
        'multi_scale': [0.5, 1.25],
        'trans_config': {# Basic Augment
                          'degrees': 0.0,
                          'translate': 0.2,
                          'scale': 0.9,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          # Mosaic & Mixup
                          'mosaic_prob': 1.0,
                          'mixup_prob': 0.15,
                          'mosaic_type': 'yolov5_mosaic',
                          'mixup_type': 'yolov5_mixup',
                          'mixup_scale': [0.5, 1.5],
                          },
        # model
        'backbone': 'elannet_large',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'yolov7_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'nbranch': 4.0,       # number of branch in ELANBlockFPN
        'depth': 1.0,         # depth factor of each branch in ELANBlockFPN
        'width': 1.0,         # width factor of channel in FPN
        # head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # matcher
        'matcher': {'soft_center_radius': 2.5,
                    'topk_candicate': 10,
                    'iou_weight': 2.0},
        # loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # training configuration
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',        # optional: sgd, adam, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        # model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,               # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolo_free_v2_huge': {
        # input
        'multi_scale': [0.5, 1.25],
        'trans_config': {# Basic Augment
                          'degrees': 0.0,
                          'translate': 0.2,
                          'scale': 0.9,
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          # Mosaic & Mixup
                          'mosaic_prob': 1.0,
                          'mixup_prob': 0.15,
                          'mosaic_type': 'yolov5_mosaic',
                          'mixup_type': 'yolov5_mixup',
                          'mixup_scale': [0.5, 1.5],
                          },
        # model
        'backbone': 'elannet_huge',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'yolov7_pafpn',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        'nbranch': 4.0,        # number of branch in ELANBlockFPN
        'depth': 2.0,          # depth factor of each branch in ELANBlockFPN
        'width': 1.25,         # width factor of channel in FPN
        # head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # matcher
        'matcher': {'soft_center_radius': 2.5,
                    'topk_candicate': 10,
                    'iou_weight': 2.0},
        # loss weight
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_box_weight': 5.0,
        # training configuration
        'no_aug_epoch': 20,
        # optimizer
        'optimizer': 'sgd',        # optional: sgd, adam, adamw
        'momentum': 0.937,         # SGD: 0.937;    AdamW: invalid
        'weight_decay': 5e-4,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        # model EMA
        'ema_decay': 0.9999,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        # lr schedule
        'scheduler': 'linear',
        'lr0': 0.01,               # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

}