from .yolo_free_config import yolo_free_config
from .yolov3_config import yolov3_config
from .yolov4_config import yolov4_config


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.version.upper()))
    
    if args.version in ['yolo_free_nano',  'yolo_free_tiny',
                        'yolo_free_small', 'yolo_free_medium',
                        'yolo_free_large', 'yolo_free_huge']:
        cfg = yolo_free_config[args.version]

    elif args.version == 'yolov3':
        cfg = yolov3_config[args.version]

    elif args.version == 'yolov4':
        cfg = yolov4_config[args.version]

    return cfg
