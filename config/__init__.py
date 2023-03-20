from .yolov8_config import yolov8_config
from .yolox_config import yolox_config
from .yolo_free_v2_config import yolo_free_v2_config


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.version.upper()))
    
    if args.version in ['yolov8_nano', 'yolov8_small', 'yolov8_medium',
                        'yolov8_large', 'yolov8_huge',     # P5
                        ]:
        cfg = yolov8_config[args.version]

    elif args.version in ['yolox_nano', 'yolox_small', 'yolox_medium',
                          'yolox_large', 'yolox_huge'
                          ]:
        cfg = yolox_config[args.version]

    elif args.version in ['yolo_free_v2_nano', 'yolo_free_v2_small',
                          'yolo_free_v2_medium', 'yolo_free_v2_large',
                          'yolo_free_v2_huge', # P5
                          ]:
        cfg = yolo_free_v2_config[args.version]

    return cfg
