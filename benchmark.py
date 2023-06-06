import argparse
import numpy as np
import time
import os
import torch

# load dataset
from dataset.build import build_transform
from dataset.coco import COCODataset, coco_class_index, coco_class_labels

# load some utils
from utils.misc import compute_flops
from utils.misc import load_weight

from config import build_dataset_config, build_model_config
from models.detectors import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='FreeYOLOv2')
    # Model
    parser.add_argument('-m', '--model', default='yolo_free_v2_large', type=str,
                        help='build yolo_free_v2')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='NMS threshold')
    parser.add_argument("--deploy", action="store_true", default=False,
                        help="deploy mode or not")
    parser.add_argument('--fuse_repconv', action='store_true', default=False,
                        help='fuse RepConv')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    # data root
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')

    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the min size of input image')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    # cuda
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')

    return parser.parse_args()


def test(net, device, img_size, testset, transform):
    # Step-1: Compute FLOPs and Params
    compute_flops(model=net, 
                     img_size=img_size, 
                     device=device)

    # Step-2: Compute FPS
    num_images = 2002
    total_time = 0
    count = 0
    with torch.no_grad():
        for index in range(num_images):
            if index % 500 == 0:
                print('Testing image {:d}/{:d}....'.format(index+1, num_images))
            image, _ = testset.pull_image(index)

            orig_h, orig_w, _ = image.shape

            # prepare
            x, _, deltas = transform(image)
            x = x.unsqueeze(0).to(device) / 255.

            # star time
            torch.cuda.synchronize()
            start_time = time.perf_counter()    

            # inference + post-process
            bboxes, scores, labels = model(x)
            
            # rescale
            img_h, img_w = x.shape[-2:]
            bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / (img_w - deltas[0]) * orig_w
            bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / (img_h - deltas[1]) * orig_h

            # clip bbox
            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

            # end time
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            # print("detection time used ", elapsed, "s")
            if index > 1:
                total_time += elapsed
                count += 1
            
        print('- FPS :', 1.0 / (total_time / count))



if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    data_dir = os.path.join(args.root, 'COCO')
    class_names = coco_class_labels
    class_indexs = coco_class_index
    num_classes = 80
    dataset = COCODataset(
                data_dir=data_dir,
                image_set='val2017',
                img_size=args.img_size)

    # Dataset & Model Config
    data_cfg = build_dataset_config(args)
    model_cfg = build_model_config(args)

    # build model
    model = build_model(args=args, 
                        cfg=model_cfg,
                        device=device, 
                        num_classes=num_classes, 
                        trainable=False)

    # load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn, args.fuse_repconv)

    # transform
    val_transform, _ = build_transform(
        args=args,
        max_stride=model_cfg['max_stride'],
        is_train=False
        )

    # run
    print("================= DETECT =================")
    test(net=model, 
        img_size=args.img_size,
        device=device, 
        testset=dataset,
        transform=val_transform
        )
