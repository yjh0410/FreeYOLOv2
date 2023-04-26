import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import os
import cv2
import math
import numpy as np
from copy import deepcopy
from thop import profile

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.crowdhuman_evaluator import CrowdHumanEvaluator
from evaluator.widerface_evaluator import WiderFaceEvaluator
from evaluator.ourdataset_evaluator import OurDatasetEvaluator

from dataset.voc import VOCDetection, VOC_CLASSES
from dataset.coco import COCODataset, coco_class_index, coco_class_labels
from dataset.widerface import WiderFaceDataset, widerface_class_labels
from dataset.crowdhuman import CrowdHumanDataset, crowd_class_labels
from dataset.ourdataset import OurDataset, our_class_labels
from dataset.transforms import build_transform

from models.yolo_free_v2.yolo_free_v2_basic import RepConv


# ---------------------------- For Dataset ----------------------------
## build dataset
def build_dataset(cfg, args, device, is_train=False):
    # transform
    print('==============================')
    print('TrainTransforms: {}'.format(cfg['trans_config']))
    trans_config = cfg['trans_config']
    train_transform = build_transform(args.img_size, trans_config, max(cfg['stride']), True)
    val_transform = build_transform(args.img_size, trans_config, max(cfg['stride']), False)
    
    # dataset params
    transform = train_transform if is_train else None
    trans_config=cfg['trans_config'] if is_train else None

    # mosaic prob.
    if args.mosaic is not None:
        trans_config['mosaic_prob']=args.mosaic if is_train else 0.0
    else:
        trans_config['mosaic_prob']=trans_config['mosaic_prob'] if is_train else 0.0

    # mixup prob.
    if args.mixup is not None:
        trans_config['mixup_prob']=args.mixup if is_train else 0.0
    else:
        trans_config['mixup_prob']=trans_config['mixup_prob']  if is_train else 0.0

    # dataset
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        class_names = VOC_CLASSES
        class_indexs = None

        # dataset
        dataset = VOCDetection(
            img_size=args.img_size,
            data_dir=data_dir,
            image_sets=[('2007', 'trainval'), ('2012', 'trainval')] if is_train else [('2007', 'test')],
            transform=transform,
            trans_config=trans_config
            )
        # evaluator
        if is_train:
            evaluator = VOCAPIEvaluator(
                data_dir=data_dir,
                device=device,
                transform=val_transform)
        else:
            evaluator = None

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        class_names = coco_class_labels
        class_indexs = coco_class_index

        # dataset
        dataset = COCODataset(
            img_size=args.img_size,
            data_dir=data_dir,
            image_set='train2017' if is_train else 'val2017',
            transform=transform,
            trans_config=trans_config
            )
        # evaluator
        if is_train:
            evaluator = COCOAPIEvaluator(
                data_dir=data_dir,
                device=device,
                transform=val_transform
                )
        else:
            evaluator = None

    elif args.dataset == 'widerface':
        data_dir = os.path.join(args.root, 'WiderFace')
        num_classes = 1
        class_names = widerface_class_labels
        class_indexs = None

        # dataset
        dataset = WiderFaceDataset(
            data_dir=data_dir,
            img_size=args.img_size,
            image_set='train' if is_train else 'val',
            transform=transform,
            trans_config=trans_config,
            )
        # evaluator
        if is_train:
            evaluator = WiderFaceEvaluator(
                data_dir=data_dir,
                device=device,
                image_set='val',
                transform=val_transform
            )
        else:
            evaluator = None

    elif args.dataset == 'crowdhuman':
        data_dir = os.path.join(args.root, 'CrowdHuman')
        num_classes = 1
        class_names = crowd_class_labels
        class_indexs = None

        # dataset
        dataset = CrowdHumanDataset(
            data_dir=data_dir,
            img_size=args.img_size,
            image_set='train' if is_train else 'val',
            transform=transform,
            trans_config=trans_config,
            )
        # evaluator
        if is_train:
            evaluator = CrowdHumanEvaluator(
                data_dir=data_dir,
                device=device,
                image_set='val',
                transform=val_transform
            )
        else:
            evaluator = None

    elif args.dataset == 'ourdataset':
        data_dir = os.path.join(args.root, 'OurDataset')
        class_names = our_class_labels
        num_classes = len(our_class_labels)
        class_indexs = None

        # dataset
        dataset = OurDataset(
            data_dir=data_dir,
            img_size=args.img_size,
            image_set='train' if is_train else 'val',
            transform=transform,
            trans_config=trans_config,
            )
        # evaluator
        if is_train:
            evaluator = OurDatasetEvaluator(
                data_dir=data_dir,
                device=device,
                image_set='val',
                transform=val_transform
            )
        else:
            evaluator = None

    else:
        print('unknow dataset !!')
        exit(0)

    print('==============================')
    print('Dataset name: {}'.format(args.dataset))
    print('Dataset size: {}'.format(len(dataset)))

    return dataset, (num_classes, class_names, class_indexs), evaluator

## build dataloader
def build_dataloader(args, dataset, batch_size, collate_fn=None):
    # distributed
    if args.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    dataloader = DataLoader(dataset, batch_sampler=batch_sampler_train,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    return dataloader
    
## collate_fn for dataloader
class CollateFunc(object):
    def __call__(self, batch):
        targets = []
        images = []

        for sample in batch:
            image = sample[0]
            target = sample[1]

            images.append(image)
            targets.append(target)

        images = torch.stack(images, 0) # [B, C, H, W]

        return images, targets


# ---------------------------- For Model ----------------------------
## fuse Conv & BN layer
def fuse_conv_bn(module):
    """Recursively fuse conv and bn in a module.
    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.
    Args:
        module (nn.Module): Module to be fused.
    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None
    
    def _fuse_conv_bn(conv, bn):
        """Fuse conv and bn into one module.
        Args:
            conv (nn.Module): Conv to be fused.
            bn (nn.Module): BN to be fused.
        Returns:
            nn.Module: Fused module.
        """
        conv_w = conv.weight
        conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
            bn.running_mean)

        factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv.weight = nn.Parameter(conv_w *
                                factor.reshape([conv.out_channels, 1, 1, 1]))
        conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
        return conv
    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module

## load trained weight
def load_weight(model, path_to_ckpt, fuse_cbn=False, fuse_repconv=False):
    # check ckpt file
    if path_to_ckpt is None:
        print('no weight file ...')

        return model

    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    checkpoint_state_dict = checkpoint.pop("model")
    model.load_state_dict(checkpoint_state_dict)

    print('Finished loading model!')

    # fuse repconv
    if fuse_repconv:
        print('Fusing RepConv block ...')
        for m in model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()

    # fuse conv & bn
    if fuse_cbn:
        print('Fusing Conv & BN ...')
        model = fuse_conv_bn(model)

    return model

## replace module
def replace_module(module, replaced_module_type, new_module_type, replace_func=None) -> nn.Module:
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model

## compute FLOPs & Parameters
def compute_flops(model, img_size, device):
    x = torch.randn(1, 3, img_size, img_size).to(device)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))

## Model EMA
class ModelEMA(object):
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(self.de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)


    def is_parallel(self, model):
        # Returns True if model is of type DP or DDP
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


    def de_parallel(self, model):
        # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
        return model.module if self.is_parallel(model) else model


    def copy_attr(self, a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)


    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = self.de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'


    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        self.copy_attr(self.ema, model, include, exclude)


# ---------------------------- NMS ----------------------------
## basic NMS
def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

## class-agnostic NMS 
def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

## class-aware NMS 
def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep = np.zeros(len(bboxes), dtype=np.int)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

## multi-class NMS 
def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)


# ---------------------------- Processor for Deployment ----------------------------
## Pre-processer
class PreProcessor(object):
    def __init__(self, img_size):
        self.img_size = img_size
        self.input_size = [img_size, img_size]
        

    def __call__(self, image, swap=(2, 0, 1)):
        """
        Input:
            image: (ndarray) [H, W, 3] or [H, W]
            formar: color format
        """
        if len(image.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3), np.float32) * 114.
        else:
            padded_img = np.ones(self.input_size, np.float32) * 114.
        # resize
        orig_h, orig_w = image.shape[:2]
        r = min(self.input_size[0] / orig_h, self.input_size[1] / orig_w)
        resize_size = (int(orig_w * r), int(orig_h * r))
        if r != 1:
            resized_img = cv2.resize(image, resize_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = image

        # padding
        padded_img[:resized_img.shape[0], :resized_img.shape[1]] = resized_img
        
        # [H, W, C] -> [C, H, W]
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)


        return padded_img, r

## Post-processer
class PostProcessor(object):
    def __init__(self, img_size, strides, num_classes, conf_thresh=0.15, nms_thresh=0.5, use_dfl=False):
        self.img_size = img_size
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.strides = strides
        self.use_dfl = use_dfl

        # generate anchors
        self.anchors, self.expand_strides = self.generate_anchors()


    def generate_anchors(self):
        """
            fmp_size: (List) [H, W]
        """
        all_anchors = []
        all_expand_strides = []
        for stride in self.strides:
            # generate grid cells
            fmp_h, fmp_w = self.img_size // stride, self.img_size // stride
            anchor_x, anchor_y = np.meshgrid(np.arange(fmp_w), np.arange(fmp_h))
            # [H, W, 2]
            anchor_xy = np.stack([anchor_x, anchor_y], axis=-1)
            shape = anchor_xy.shape[:2]
            # [H, W, 2] -> [HW, 2]
            anchor_xy = (anchor_xy.reshape(-1, 2) + 0.5) * stride
            all_anchors.append(anchor_xy)

            # expanded stride
            strides = np.full((*shape, 1), stride)
            all_expand_strides.append(strides.reshape(-1, 1))

        anchors = np.concatenate(all_anchors, axis=0)
        expand_strides = np.concatenate(all_expand_strides, axis=0)

        return anchors, expand_strides
        

    def decode_boxes(self, anchors, pred_regs):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # tlbr -> xyxy
        pred_x1y1 = anchors - pred_regs[..., :2] * self.expand_strides
        pred_x2y2 = anchors + pred_regs[..., 2:] * self.expand_strides
        pred_box = np.concatenate([pred_x1y1, pred_x2y2], axis=-1)

        return pred_box


    def __call__(self, predictions):
        """
        Input:
            predictions: (ndarray) [n_anchors_all, 4+1+C]
        """
        reg_preds = predictions[..., :4]
        scores = predictions[..., 4:]

        # scores & labels
        labels = np.argmax(scores, axis=1)                      # [M,]
        scores = scores[(np.arange(scores.shape[0]), labels)]   # [M,]

        # bboxes
        bboxes = self.decode_boxes(self.anchors, reg_preds)     # [M, 4]    

        # thresh
        keep = np.where(scores > self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, True)

        return bboxes, scores, labels
