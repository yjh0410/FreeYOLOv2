# ------------------------ Model Config ------------------------
from .yolo_free_v2_config import yolo_free_v2_cfg


def build_model_config(args):
    if args.model in ['yolo_free_v2_pico', 'yolo_free_v2_nano', 'yolo_free_v2_small',
                      'yolo_free_v2_medium', 'yolo_free_v2_large', 'yolo_free_v2_huge']:
        cfg = yolo_free_v2_cfg[args.model]

    print('==============================')
    print('Model Config: {} \n'.format(cfg))

    return cfg


# ------------------------ Dataset Config ------------------------
from .dataset_config import dataset_cfg


def build_dataset_config(args):
    if args.dataset in ['coco', 'coco-val', 'coco-test']:
        cfg = dataset_cfg['coco']
    else:
        cfg = dataset_cfg[args.dataset]

    print('==============================')
    print('Dataset Config: {} \n'.format(cfg))

    return cfg
