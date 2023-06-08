import argparse
import os

from copy import deepcopy
import torch

from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.crowdhuman_evaluator import CrowdHumanEvaluator
from evaluator.widerface_evaluator import WiderFaceEvaluator
from evaluator.ourdataset_evaluator import OurDatasetEvaluator

# load transform
from dataset.build import build_transform

# load some utils
from utils.misc import load_weight
from utils.misc import compute_flops

from config import build_dataset_config, build_model_config
from models.detectors import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='FreeYOLOv2')
    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')

    # model
    parser.add_argument('-m', '--model', default='yolo_free_v2_large', type=str,
                        help='build yolo_free_v2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk candidates for testing')
    parser.add_argument("--deploy", action="store_true", default=False,
                        help="deploy mode or not")
    parser.add_argument('--fuse_repconv', action='store_true', default=False,
                        help='fuse RepConv')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')

    return parser.parse_args()



def voc_test(model, data_dir, device, transform):
    evaluator = VOCAPIEvaluator(data_dir=data_dir,
                                device=device,
                                transform=transform,
                                display=True)

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, data_dir, device, transform, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=True,
                        transform=transform)

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=False,
                        transform=transform)

    # COCO evaluation
    evaluator.evaluate(model)


def crowdhuman_test(model, data_dir, device, transform):
    evaluator = CrowdHumanEvaluator(
        data_dir=data_dir,
        device=device,
        image_set='val',
        transform=transform)

    # WiderFace evaluation
    evaluator.evaluate(model)


def widerface_test(model, data_dir, device, transform):
    evaluator = WiderFaceEvaluator(
        data_dir=data_dir,
        device=device,
        image_set='val',
        transform=transform)

    # WiderFace evaluation
    evaluator.evaluate(model)


def our_test(model, data_dir, device, transform):
    evaluator = OurDatasetEvaluator(
        data_dir=data_dir,
        device=device,
        image_set='val',
        transform=transform)

    # WiderFace evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Dataset & Model Config
    data_cfg = build_dataset_config(args)
    model_cfg = build_model_config(args)
    data_dir = os.path.join(args.root, data_cfg['data_name'])
    num_classes = data_cfg['num_classes']

    # build model
    model = build_model(args, model_cfg, device, num_classes, False)

    # load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn, args.fuse_repconv)
    model.to(device).eval()

    # compute FLOPs and Params
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    compute_flops(
        model=model_copy,
        img_size=args.img_size, 
        device=device)
    del model_copy

    # transform
    val_transform, _ = build_transform(
        args=args,
        max_stride=model_cfg['max_stride'],
        is_train=False
        )

    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, data_dir, device, val_transform)
        elif args.dataset == 'coco-val':
            coco_test(model, data_dir, device, val_transform, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, data_dir, device, val_transform, test=True)
        elif args.dataset == 'widerface':
            widerface_test(model, data_dir, device, val_transform)
        elif args.dataset == 'crowdhuman':
            crowdhuman_test(model, data_dir, device, val_transform)
        elif args.dataset == 'ourdataset':
            our_test(model, data_dir, device, val_transform)
