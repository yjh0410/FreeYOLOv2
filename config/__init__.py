from .yolo_free_v2_config import yolo_free_v2_config


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

    return cfg
