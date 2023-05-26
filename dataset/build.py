import os

from dataset.voc import VOCDetection
from dataset.coco import COCODataset
from dataset.widerface import WiderFaceDataset
from dataset.crowdhuman import CrowdHumanDataset
from dataset.ourdataset import OurDataset

from dataset.transforms import TrainTransforms, ValTransforms


# ------------------------------ Dataset ------------------------------
def build_dataset(args, data_cfg, trans_config, transform, is_train=False):
    # Basic parameters
    data_dir = os.path.join(args.root, data_cfg['data_name'])
    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    class_indexs = data_cfg['class_indexs']
    dataset_info = {
        'num_classes': num_classes,
        'class_names': class_names,
        'class_indexs': class_indexs
    }

    # Build dataset class
    ## VOC dataset
    if args.dataset == 'voc':
        dataset = VOCDetection(
            img_size=args.img_size,
            data_dir=data_dir,
            image_sets=[('2007', 'trainval'), ('2012', 'trainval')] if is_train else [('2007', 'test')],
            transform=transform,
            trans_config=trans_config
            )
    ## COCO dataset
    elif args.dataset == 'coco':
        dataset = COCODataset(
            img_size=args.img_size,
            data_dir=data_dir,
            image_set='train2017' if is_train else 'val2017',
            transform=transform,
            trans_config=trans_config
            )
    ## WiderFace dataset
    elif args.dataset == 'widerface':
        dataset = WiderFaceDataset(
            data_dir=data_dir,
            img_size=args.img_size,
            image_set='train' if is_train else 'val',
            transform=transform,
            trans_config=trans_config,
            )
    ## CrowdHuman dataset
    elif args.dataset == 'crowdhuman':
        dataset = CrowdHumanDataset(
            data_dir=data_dir,
            img_size=args.img_size,
            image_set='train' if is_train else 'val',
            transform=transform,
            trans_config=trans_config,
            )
    ## Custom dataset
    elif args.dataset == 'ourdataset':
        dataset = OurDataset(
            data_dir=data_dir,
            img_size=args.img_size,
            image_set='train' if is_train else 'val',
            transform=transform,
            trans_config=trans_config,
            )

    return dataset, dataset_info


# ------------------------------ Transform ------------------------------
def build_transform(args, trans_config=None, max_stride=32, is_train=False):
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))
    
    # Modify trans_config
    if trans_config is not None:
        ## mosaic prob.
        if args.mosaic is not None:
            trans_config['mosaic_prob']=args.mosaic if is_train else 0.0
        else:
            trans_config['mosaic_prob']=trans_config['mosaic_prob'] if is_train else 0.0
        ## mixup prob.
        if args.mixup is not None:
            trans_config['mixup_prob']=args.mixup if is_train else 0.0
        else:
            trans_config['mixup_prob']=trans_config['mixup_prob']  if is_train else 0.0
    # Transform
    if is_train:
        transform = TrainTransforms(
            img_size=args.img_size,
            trans_config=trans_config,
            min_box_size=args.min_box_size
            )
    else:
        transform = ValTransforms(
            img_size=args.img_size,
            max_stride=max_stride
            )

    return transform, trans_config
