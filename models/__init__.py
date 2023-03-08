#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from .yolo_free_v2.build import build_yolo_free_v2
from .yolo_free_vx.build import build_yolo_free_vx
from .yolov3_ex.build import build_yolov3
from .yolov4_ex.build import build_yolov4


# build object detector
def build_model(args, 
                cfg,
                device, 
                num_classes=80, 
                trainable=False):
    # detector    
    if args.version in ['yolo_free_v2_pico', 'yolo_free_v2_nano',  'yolo_free_v2_tiny',
                        'yolo_free_v2_small', 'yolo_free_v2_medium', 'yolo_free_v2_large',
                        'yolo_free_v2_huge',     # P5
                        'yolo_free_v2_pico_p6', 'yolo_free_v2_nano_p6',  'yolo_free_v2_tiny_p6',
                        'yolo_free_v2_small_p6', 'yolo_free_v2_medium_p6', 'yolo_free_v2_large_p6',
                        'yolo_free_v2_huge_p6',  # P6
                        'yolo_free_v2_pico_p7', 'yolo_free_v2_nano_p7', 'yolo_free_v2_tiny_p7',
                        'yolo_free_v2_small_p7', 'yolo_free_v2_medium_p7', 'yolo_free_v2_large_p7',
                        'yolo_free_v2_huge_p7'   # P7
                        ]:
        model, criterion = build_yolo_free_v2(
            args, cfg, device, num_classes, trainable)

    elif args.version in ['yolo_free_vx_pico', 'yolo_free_vx_nano', 'yolo_free_vx_tiny',
                          'yolo_free_vx_small', 'yolo_free_vx_medium', 'yolo_free_vx_large',
                          'yolo_free_vx_huge', # P5
                          ]:
        model, criterion = build_yolo_free_vx(
            args, cfg, device, num_classes, trainable)
        
    elif args.version == 'yolov3_ex':
        model, criterion = build_yolov3(
            args, cfg, device, num_classes, trainable)

    elif args.version == 'yolov4_ex':
        model, criterion = build_yolov4(
            args, cfg, device, num_classes, trainable)
        
    if trainable:
        # Load pretrained weight
        if args.pretrained is not None:
            print('Loading COCO pretrained weight ...')
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = model.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                        print(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print(k)

            model.load_state_dict(checkpoint_state_dict, strict=False)

        # keep training
        if args.resume is not None:
            print('keep training: ', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)

        return model, criterion

    else:      
        return model
