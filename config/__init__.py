from .yolov8_config import yolov8_config
from .yolo_free_v2_config import yolo_free_v2_config


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.version.upper()))
    # YOLOv8
    if args.version in ['yolov8_nano', 'yolov8_small', 'yolov8_medium',
                        'yolov8_large', 'yolov8_huge',     # P5
                        ]:
        cfg = yolov8_config[args.version]
    # FreeYOLOv2
    elif args.version in ['yolo_free_v2_nano', 'yolo_free_v2_small',
                          'yolo_free_v2_medium', 'yolo_free_v2_large',
                          'yolo_free_v2_huge', # P5
                          ]:
        cfg = yolo_free_v2_config[args.version]

    return cfg
