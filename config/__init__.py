from .yolov8_config import yolov8_config
from .yolo_free_vx_config import yolo_free_vx_config


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.version.upper()))
    
    if args.version in ['yolov8_nano', 'yolov8_small', 'yolov8_medium',
                        'yolov8_large', 'yolov8_huge',     # P5
                        ]:
        cfg = yolov8_config[args.version]

    elif args.version in ['yolo_free_vx_pico', 'yolo_free_vx_nano', 'yolo_free_vx_tiny',
                          'yolo_free_vx_small', 'yolo_free_vx_medium', 'yolo_free_vx_large',
                          'yolo_free_vx_huge', # P5
                          ]:
        cfg = yolo_free_vx_config[args.version]
        
    return cfg
