from .yolo_free_v2_config import yolo_free_v2_config
from .yolov3_e_config import yolov3_e_config
from .yolov4_e_config import yolov4_e_config


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.version.upper()))
    
    if args.version in ['yolo_free_v2_nano',  'yolo_free_v2_tiny',
                        'yolo_free_v2_small', 'yolo_free_v2_medium',
                        'yolo_free_v2_large', 'yolo_free_v2_huge',
                        'yolo_free_v2_nano_p6', 'yolo_free_v2_tiny_p6',
                        'yolo_free_v2_small_p6', 'yolo_free_v2_medium_p6',
                        'yolo_free_v2_large_p6', 'yolo_free_v2_huge_p6']:
        cfg = yolo_free_v2_config[args.version]

    elif args.version == 'yolov3_e':
        cfg = yolov3_e_config[args.version]

    elif args.version == 'yolov4_e':
        cfg = yolov4_e_config[args.version]

    return cfg