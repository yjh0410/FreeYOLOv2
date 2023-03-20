#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from .yolov8.build import build_yolov8
from .yolox.build import build_yolox
from .yolo_free_v2.build import build_yolo_free_v2


# build object detector
def build_model(args, 
                cfg,
                device, 
                num_classes=80, 
                trainable=False):
    # detector    
    if args.version in ['yolov8_nano', 'yolov8_small', 'yolov8_medium',
                        'yolov8_large', 'yolov8_huge',     # P5
                        ]:
        model, criterion = build_yolov8(
            args, cfg, device, num_classes, trainable)

    elif args.version in ['yolox_nano', 'yolox_small', 'yolox_medium',
                          'yolox_large', 'yolox_huge']:
        model, criterion = build_yolox(
            args, cfg, device, num_classes, trainable)

    elif args.version in ['yolo_free_v2_pico', 'yolo_free_v2_nano', 'yolo_free_v2_tiny',
                          'yolo_free_v2_small', 'yolo_free_v2_medium', 'yolo_free_v2_large',
                          'yolo_free_v2_huge', # P5
                        ]:
        model, criterion = build_yolo_free_v2(
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
