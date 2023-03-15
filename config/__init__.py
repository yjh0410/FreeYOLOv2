from .yolo_free_v2_config import yolo_free_v2_config
from .yolo_free_vx_config import yolo_free_vx_config


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.version.upper()))
    
    if args.version in ['yolo_free_v2_pico', 'yolo_free_v2_nano',  'yolo_free_v2_tiny',
                        'yolo_free_v2_small', 'yolo_free_v2_medium', 'yolo_free_v2_large',
                        'yolo_free_v2_huge',     # P5
                        ]:
        cfg = yolo_free_v2_config[args.version]

    elif args.version in ['yolo_free_vx_pico', 'yolo_free_vx_nano', 'yolo_free_vx_tiny',
                          'yolo_free_vx_small', 'yolo_free_vx_medium', 'yolo_free_vx_large',
                          'yolo_free_vx_huge', # P5
                          ]:
        cfg = yolo_free_vx_config[args.version]
        
    return cfg
