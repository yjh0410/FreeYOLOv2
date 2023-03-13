from .yolo_free_v2_config import yolo_free_v2_config
from .yolov3_ex_config import yolov3_ex_config
from .yolov4_ex_config import yolov4_ex_config


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.version.upper()))
    
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
        cfg = yolo_free_v2_config[args.version]

    elif args.version == 'yolov3_ex':
        cfg = yolov3_ex_config[args.version]

    elif args.version == 'yolov4_ex':
        cfg = yolov4_ex_config[args.version]

    return cfg
