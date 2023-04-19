from .yolo_free_v2_config import yolo_free_v2_cfg


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.model.upper()))
    if args.model in ['yolo_free_v2_nano', 'yolo_free_v2_tiny',
                      'yolo_free_v2_large', 'yolo_free_v2_huge', # P5
                     ]:
        cfg = yolo_free_v2_cfg[args.model]

    return cfg
