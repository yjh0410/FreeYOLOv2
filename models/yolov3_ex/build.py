import torch
from .loss import build_criterion
from .yolov3_ex import YOLOv3_EX


# build object detector
def build_yolov3(args, cfg, device, num_classes=80, trainable=False):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    print('==============================')
    print('Model Configuration: \n', cfg)
    
    model = YOLOv3_EX(
        cfg=cfg,
        device=device, 
        num_classes=num_classes,
        trainable=trainable,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        topk=args.topk,
        no_decode=args.no_decode
        )

    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)
    return model, criterion