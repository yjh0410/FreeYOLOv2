# FreeYOLOv2 config


yolo_free_v2_cfg = {
    'yolo_free_v2_pico': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': True,
        'width': 0.25,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck: SPP
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': True,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'elanblock',
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_depthwise': True,
        ## Head
        'head': 'decoupled_head',
        'head_act': 'silu',
        'head_norm': 'BN',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': True,
        'reg_max': 16,
        # ---------------- Preprocess ----------------
        'multi_scale': [0.5, 1.5],
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
                          'mosaic_9x_prob': 0.0,
                          'mixup_prob': 0.0,
                          },
        # ---------------- Assignment config ----------------
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_weights': {'bce':{
                            'loss_cls_weight': 0.5,
                            'loss_iou_weight': 7.5,
                            'loss_dfl_weight': 1.5
                            },
                         'vfl':{
                            'loss_cls_weight': 1.0,
                            'loss_iou_weight': 2.0,
                            'loss_dfl_weight': 1.0
                            },
                         },
        # ---------------- Train config ----------------
        'trainer': 'yolo',
        },

    'yolo_free_v2_nano': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.25,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck: SPP
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'elanblock',
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
        'reg_max': 16,
        # ---------------- Preprocess ----------------
        'multi_scale': [0.5, 1.5],
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
                          'mosaic_9x_prob': 0.0,
                          'mixup_prob': 0.0,
                          },
        # ---------------- Assignment config ----------------
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_weights': {'bce':{
                            'loss_cls_weight': 0.5,
                            'loss_iou_weight': 7.5,
                            'loss_dfl_weight': 1.5
                            },
                         'vfl':{
                            'loss_cls_weight': 1.0,
                            'loss_iou_weight': 2.0,
                            'loss_dfl_weight': 1.0
                            },
                         },
        # ---------------- Train config ----------------
        'trainer': 'yolo',
        },

    'yolo_free_v2_small': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.50,
        'depth': 0.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck: SPP
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'elanblock',
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
        'reg_max': 16,
        # ---------------- Preprocess ----------------
        'multi_scale': [0.5, 1.5],
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
                          'mosaic_9x_prob': 0.0,
                          'mixup_prob': 0.0,
                          },
        # ---------------- Assignment config ----------------
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_weights': {'bce':{
                            'loss_cls_weight': 0.5,
                            'loss_iou_weight': 7.5,
                            'loss_dfl_weight': 1.5
                            },
                         'vfl':{
                            'loss_cls_weight': 1.0,
                            'loss_iou_weight': 2.0,
                            'loss_dfl_weight': 1.0
                            },
                         },
        # ---------------- Train config ----------------
        'trainer': 'yolo',
        },

    'yolo_free_v2_medium': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 0.75,
        'depth': 0.67,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck: SPP
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'elanblock',
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
        'reg_max': 16,
        # ---------------- Preprocess ----------------
        'multi_scale': [0.5, 1.5],
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
                          'mosaic_9x_prob': 0.2,
                          'mixup_prob': 0.1,
                          },
        # ---------------- Assignment config ----------------
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_weights': {'bce':{
                            'loss_cls_weight': 0.5,
                            'loss_iou_weight': 7.5,
                            'loss_dfl_weight': 1.5
                            },
                         'vfl':{
                            'loss_cls_weight': 1.0,
                            'loss_iou_weight': 2.0,
                            'loss_dfl_weight': 1.0
                            },
                         },
        # ---------------- Train config ----------------
        'trainer': 'yolo',
        },

    'yolo_free_v2_large': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.0,
        'depth': 1.0,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck: SPP
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        ## Neck: PaFPN
        'fpn': 'yolo_pafpn',
        'fpn_reduce_layer': 'Conv',
        'fpn_downsample_layer': 'Conv',
        'fpn_core_block': 'elanblock',
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
        'reg_max': 16,
        # ---------------- Preprocess ----------------
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
                          'mosaic_9x_prob': 0.2,
                          'mixup_prob': 0.15,
                          },
        # ---------------- Assignment config ----------------
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_weights': {'bce':{
                            'loss_cls_weight': 0.5,
                            'loss_iou_weight': 7.5,
                            'loss_dfl_weight': 1.5
                            },
                         'vfl':{
                            'loss_cls_weight': 1.0,
                            'loss_iou_weight': 2.0,
                            'loss_dfl_weight': 1.0
                            },
                         },
        # ---------------- Train config ----------------
        'trainer': 'yolo',
        },

    'yolo_free_v2_huge': {
        # ---------------- Model config ----------------
        ## Backbone
        'backbone': 'elannet',
        'pretrained': True,
        'bk_act': 'silu',
        'bk_norm': 'BN',
        'bk_dpw': False,
        'width': 1.25,
        'depth': 1.34,
        'stride': [8, 16, 32],  # P3, P4, P5
        'max_stride': 32,
        ## Neck: SPP
        'neck': 'sppf',
        'neck_expand_ratio': 0.5,
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
                          'scale': 0.9,
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
        'matcher': {'topk': 10,
                    'alpha': 0.5,
                    'beta': 6.0},
        # ---------------- Loss config ----------------
        ## loss weight
        'cls_loss': 'bce', # vfl (optional)
        'loss_weights': {'bce':{
                            'loss_cls_weight': 0.5,
                            'loss_iou_weight': 7.5,
                            'loss_dfl_weight': 1.5
                            },
                         'vfl':{
                            'loss_cls_weight': 1.0,
                            'loss_iou_weight': 2.0,
                            'loss_dfl_weight': 1.0
                            },
                         },
        # ---------------- Train config ----------------
        'trainer': 'yolo',
        },

}