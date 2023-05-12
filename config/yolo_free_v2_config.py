# yolo-free config


yolo_free_v2_cfg = {
    # ----------------------- P5 Model -----------------------
    'yolo_free_v2_nano': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elan_cspnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.25,
        'depth': 0.34,
        'ratio': 2.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## Neck: SPP
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'ELAN_CSPBlock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ---------------- Preprocess ----------------
        'multi_scale': [0.5, 1.25],
        'trans_config': {# Basic Augment
                          'degrees': 0.0,
                          'translate': 0.1,
                          'scale': [0.5, 2.0],
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          # Mosaic & Mixup
                          'mosaic_prob': 1.0,
                          'mosaic_9x_prob': 0.2,
                          'mixup_prob': 0.05,
                          },
        # ---------------- Assignment config ----------------
        'matcher': {'soft_center_radius': 3.0,
                    'topk_candicate': 13,
                    'iou_weight': 3.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'AdamW',      # optional: SGD, AdamW
        'momentum': None,          # SGD: 0.937;    AdamW: None
        'weight_decay': 5e-2,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## model EMA
        'ema_decay': 0.9998,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## lr schedule
        'scheduler': 'linear',
        'lr0': 0.001,              # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolo_free_v2_small': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elan_cspnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.5,
        'depth': 0.34,
        'ratio': 2.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## Neck: SPP
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'ELAN_CSPBlock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ---------------- Preprocess ----------------
        'multi_scale': [0.5, 1.25],
        'trans_config': {# Basic Augment
                          'degrees': 0.0,
                          'translate': 0.2,
                          'scale': [0.1, 2.0],
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          # Mosaic & Mixup
                          'mosaic_prob': 1.0,
                          'mosaic_9x_prob': 0.2,
                          'mixup_prob': 0.15,
                          },
        # ---------------- Assignment config ----------------
        'matcher': {'soft_center_radius': 3.0,
                    'topk_candicate': 13,
                    'iou_weight': 3.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'AdamW',      # optional: SGD, AdamW
        'momentum': None,          # SGD: 0.937;    AdamW: None
        'weight_decay': 5e-2,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## model EMA
        'ema_decay': 0.9998,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## lr schedule
        'scheduler': 'linear',
        'lr0': 0.001,              # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolo_free_v2_medium': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elan_cspnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.75,
        'depth': 0.67,
        'ratio': 1.50,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## Neck: SPP
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'ELAN_CSPBlock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ---------------- Preprocess ----------------
        'multi_scale': [0.5, 1.25],
        'trans_config': {# Basic Augment
                          'degrees': 0.0,
                          'translate': 0.2,
                          'scale': [0.1, 2.0],
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          # Mosaic & Mixup
                          'mosaic_prob': 1.0,
                          'mosaic_9x_prob': 0.2,
                          'mixup_prob': 0.15,
                          },
        # ---------------- Assignment config ----------------
        'matcher': {'soft_center_radius': 3.0,
                    'topk_candicate': 13,
                    'iou_weight': 3.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'AdamW',      # optional: SGD, AdamW
        'momentum': None,          # SGD: 0.937;    AdamW: None
        'weight_decay': 5e-2,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## model EMA
        'ema_decay': 0.9998,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## lr schedule
        'scheduler': 'linear',
        'lr0': 0.001,              # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolo_free_v2_large': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elan_cspnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.0,
        'depth': 1.0,
        'ratio': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## Neck: SPP
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'ELAN_CSPBlock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ---------------- Preprocess ----------------
        'multi_scale': [0.5, 1.25],
        'trans_config': {# Basic Augment
                          'degrees': 0.0,
                          'translate': 0.2,
                          'scale': [0.1, 2.0],
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          # Mosaic & Mixup
                          'mosaic_prob': 1.0,
                          'mosaic_9x_prob': 0.2,
                          'mixup_prob': 0.15,
                          },
        # ---------------- Assignment config ----------------
        'matcher': {'soft_center_radius': 3.0,
                    'topk_candicate': 13,
                    'iou_weight': 3.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'AdamW',      # optional: SGD, AdamW
        'momentum': None,          # SGD: 0.937;    AdamW: None
        'weight_decay': 5e-2,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## model EMA
        'ema_decay': 0.9998,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## lr schedule
        'scheduler': 'linear',
        'lr0': 0.001,              # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

    'yolo_free_v2_huge': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elan_cspnet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.25,
        'depth': 1.0,
        'ratio': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        ## Neck: SPP
        'neck': 'csp_sppf',
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'ELAN_CSPBlock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': False,
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # ---------------- Preprocess ----------------
        'multi_scale': [0.5, 1.25],
        'trans_config': {# Basic Augment
                          'degrees': 0.0,
                          'translate': 0.2,
                          'scale': [0.1, 2.0],
                          'shear': 0.0,
                          'perspective': 0.0,
                          'hsv_h': 0.015,
                          'hsv_s': 0.7,
                          'hsv_v': 0.4,
                          # Mosaic & Mixup
                          'mosaic_prob': 1.0,
                          'mosaic_9x_prob': 0.2,
                          'mixup_prob': 0.2,
                          },
        # ---------------- Assignment config ----------------
        'matcher': {'soft_center_radius': 3.0,
                    'topk_candicate': 13,
                    'iou_weight': 3.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'loss_cls_weight': 1.0,
        'loss_box_weight': 2.0,
        # ---------------- Train config ----------------
        ## close strong augmentation
        'no_aug_epoch': 20,
        ## optimizer
        'optimizer': 'AdamW',      # optional: SGD, AdamW
        'momentum': None,          # SGD: 0.937;    AdamW: None
        'weight_decay': 5e-2,      # SGD: 5e-4;     AdamW: 5e-2
        'clip_grad': 10,           # SGD: 10.0;     AdamW: -1
        ## model EMA
        'ema_decay': 0.9998,       # SGD: 0.9999;   AdamW: 0.9998
        'ema_tau': 2000,
        ## lr schedule
        'scheduler': 'linear',
        'lr0': 0.001,              # SGD: 0.01;     AdamW: 0.001
        'lrf': 0.01,               # SGD: 0.01;     AdamW: 0.01
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        },

}