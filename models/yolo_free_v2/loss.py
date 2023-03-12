import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import AlignSimOTA
from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        # loss
        self.cls_lossf = ClassificationLoss(cfg, reduction='none')
        self.reg_lossf = RegressionLoss(num_classes)
        # loss weight
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_reg_weight = cfg['loss_reg_weight']
        # matcher
        matcher_config = cfg['matcher']
        self.matcher = AlignSimOTA(
            num_classes=num_classes,
            center_sampling_radius=matcher_config['center_sampling_radius'],
            topk_candidate=matcher_config['topk_candicate']
            )


    def __call__(self, outputs, targets):        
        """
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        num_anchors = sum([ab.shape[0] for ab in anchors])
        # preds: [B, M, C]
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)

        # label assignment
        cls_targets = []
        box_targets = []
        fg_masks = []

        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                # There is no valid gt
                cls_target = cls_preds.new_zeros((num_anchors, self.num_classes))
                box_target = cls_preds.new_zeros((0, 4))
                fg_mask = cls_preds.new_zeros(num_anchors).bool()
            else:
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.matcher(
                    fpn_strides = fpn_strides,
                    anchors = anchors,
                    pred_cls = cls_preds[batch_idx], 
                    pred_box = box_preds[batch_idx],
                    tgt_labels = tgt_labels,
                    tgt_bboxes = tgt_bboxes
                    )
                # cls target
                cls_target = cls_preds.new_zeros((num_anchors, self.num_classes))
                gt_classes = F.one_hot(gt_matched_classes.long(), self.num_classes)
                gt_classes = gt_classes * pred_ious_this_matching.unsqueeze(-1)
                cls_target[fg_mask] = gt_classes.type_as(cls_target)
                # box target
                box_target = tgt_bboxes[matched_gt_inds]

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_foregrounds = fg_masks.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foregrounds)
        num_foregrounds = (num_foregrounds / get_world_size()).clamp(1.0)
        
        # cls loss
        cls_preds = cls_preds.view(-1, self.num_classes)
        loss_cls = self.cls_lossf(cls_preds, cls_targets)
        loss_cls = loss_cls.sum() / num_foregrounds

        # box loss
        box_preds = box_preds.view(-1, 4)
        ious = self.reg_lossf(box_preds, box_targets, fg_masks)
        loss_box = ious.sum() / num_foregrounds

        # total loss
        losses = self.loss_cls_weight * loss_cls + \
                 self.loss_reg_weight * loss_box

        loss_dict = dict(
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict
    

class ClassificationLoss(nn.Module):
    def __init__(self, cfg, reduction='none'):
        super(ClassificationLoss, self).__init__()
        self.cfg = cfg
        self.reduction = reduction
        # For VFL
        self.alpha = 0.75
        self.gamma = 2.0


    def varifocalloss(self, pred_logits, gt_score, gt_label, alpha=0.75, gamma=2.0):
        focal_weight = alpha * pred_logits.sigmoid().pow(gamma) * (1 - gt_label) + gt_score * gt_label
        with torch.cuda.amp.autocast(enabled=False):
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_logits.float(), gt_score.float(), reduction='none')
            loss = bce_loss * focal_weight

            if self.reduction == 'sum':
                loss = loss.sum()
            elif self.reduction == 'mean':
                loss = loss.mean()

        return loss


    def binary_cross_entropy(self, pred_logits, gt_score):
        loss = F.binary_cross_entropy_with_logits(
            pred_logits.float(), gt_score.float(), reduction='none')

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


    def forward(self, pred_logits, gt_score, gt_label=None):
        if self.cfg['cls_loss'] == 'vfl':
            return self.varifocalloss(pred_logits, gt_score, gt_label, self.alpha, self.gamma)
        elif self.cfg['cls_loss'] == 'bce':
            return self.binary_cross_entropy(pred_logits, gt_score)


class RegressionLoss(nn.Module):
    def __init__(self, num_classes):
        super(RegressionLoss, self).__init__()
        self.num_classes = num_classes


    def forward(self, pred_boxs, gt_boxs, fg_masks):
        """
        Input:
            pred_boxs: (Tensor) [BM, 4]
            gt_boxs: (Tensor) [BM, 4]
            fg_masks: (Tensor) [BM,]
        """
        # select positive samples mask
        num_pos = fg_masks.sum()

        if num_pos > 0:
            # iou loss
            ious = get_ious(pred_boxs[fg_masks],
                            gt_boxs,
                            box_mode="xyxy",
                            iou_type='giou')
            loss_iou = 1.0 - ious
               
        else:
            loss_iou = pred_boxs.sum() * 0.

        return loss_iou


def build_criterion(cfg, device, num_classes):
    criterion = Criterion(
        cfg=cfg,
        device=device,
        num_classes=num_classes
        )

    return criterion


if __name__ == "__main__":
    pass