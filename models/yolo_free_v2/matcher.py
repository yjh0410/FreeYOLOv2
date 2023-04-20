# ---------------------------------------------------------------------
# Copyright (c) Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------


import torch
import torch.nn.functional as F
from utils.box_ops import *


# YOLOX SimOTA
class AlignedSimOTA(object):
    """
        This code referenced to https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
    """
    def __init__(self,
                 num_classes,
                 soft_center_radius=3.0,
                 topk=13,
                 iou_weight=3.0
                 ):
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight


    @torch.no_grad()
    def __call__(self, 
                 fpn_strides, 
                 anchors, 
                 pred_cls, 
                 pred_box, 
                 gt_labels,
                 gt_bboxes):
        # [M,]
        strides = torch.cat([torch.ones_like(anchor_i[:, 0]) * stride_i
                                for stride_i, anchor_i in zip(fpn_strides, anchors)], dim=-1)
        # List[F, M, 2] -> [M, 2]
        anchors = torch.cat(anchors, dim=0)
        num_gt = len(gt_labels)

        # get inside points
        is_in_gt = self.get_in_boxes_info(gt_bboxes, anchors)
        cls_preds_ = pred_cls[is_in_gt]   # [Mp, C]
        box_preds_ = pred_box[is_in_gt]   # [Mp, 4]
        num_in_boxes_anchor = box_preds_.shape[0]

        # ----------------------------------- soft center prior -----------------------------------
        gt_center = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2.0
        distance = (anchors[is_in_gt].unsqueeze(0) - gt_center.unsqueeze(1)
                    ).pow(2).sum(-1).sqrt() / strides[is_in_gt].unsqueeze(0)  # [N, Mp]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        # ----------------------------------- regression cost -----------------------------------
        pair_wise_ious, _ = box_iou(gt_bboxes, box_preds_)  # [N, Mp]
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8) * self.iou_weight

        # ----------------------------------- classification cost -----------------------------------
        gt_cls = (
            F.one_hot(gt_labels.long(), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        ) # [N, C] -> [N, Mp, C]
        soften_gt_cls = gt_cls * pair_wise_ious.unsqueeze(-1)
        with torch.cuda.amp.autocast(enabled=False):
            # [Mp, C] -> [N, Mp, C]
            pairwise_pred_scores = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1) # [N, Mp, C]
            scale_factor = (soften_gt_cls - pairwise_pred_scores.sigmoid()).abs().pow(2.0)
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                pairwise_pred_scores, soften_gt_cls,
                reduction="none") * scale_factor
            pair_wise_cls_loss = pair_wise_cls_loss.sum(-1) # [N, Mp]
            
        del pairwise_pred_scores

        # foreground cost matrix
        cost_metrix = pair_wise_cls_loss + pair_wise_ious_loss + soft_center_prior

        (
            fg_mask,              # [num_fg,]
            assigned_labels,      # [num_fg,]
            matched_pred_ious,    # [num_fg,]
            matched_gt_inds,      # [num_fg,]
        ) = self.dynamic_k_matching(
            cost_metrix,
            pair_wise_ious,
            gt_labels,
            num_gt,
            is_in_gt
            )
        del pair_wise_cls_loss, cost_metrix, pair_wise_ious, pair_wise_ious_loss

        return (
                fg_mask,
                assigned_labels,
                matched_pred_ious,
                matched_gt_inds,
        )


    def get_in_boxes_info(self, gt_bboxes, anchors):
        """
            gt_bboxes: Tensor -> [N, 2]
            anchors:   Tensor -> [M, 2]
        """
        num_anchors = anchors.shape[0]
        num_gt = gt_bboxes.shape[0]

        # anchor center
        x_centers = anchors[:, 0]
        y_centers = anchors[:, 1]

        # [M,] -> [1, M] -> [N, M]
        x_centers = x_centers.unsqueeze(0).repeat(num_gt, 1)
        y_centers = y_centers.unsqueeze(0).repeat(num_gt, 1)

        # [N,] -> [N, 1] -> [N, M]
        gt_bboxes_l = gt_bboxes[:, 0].unsqueeze(1).repeat(1, num_anchors) # x1
        gt_bboxes_t = gt_bboxes[:, 1].unsqueeze(1).repeat(1, num_anchors) # y1
        gt_bboxes_r = gt_bboxes[:, 2].unsqueeze(1).repeat(1, num_anchors) # x2
        gt_bboxes_b = gt_bboxes[:, 3].unsqueeze(1).repeat(1, num_anchors) # y2

        b_l = x_centers - gt_bboxes_l
        b_r = gt_bboxes_r - x_centers
        b_t = y_centers - gt_bboxes_t
        b_b = gt_bboxes_b - y_centers
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        return is_in_boxes_all
    
    
    def dynamic_k_matching(
        self, 
        cost, 
        pair_wise_ious, 
        gt_classes, 
        num_gt, 
        is_in_gt
        ):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.topk, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        is_in_gt[is_in_gt.clone()] = fg_mask_inboxes
        fg_mask = is_in_gt

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        assigned_labels = gt_classes[matched_gt_inds]

        matched_pred_ious = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return fg_mask, assigned_labels, matched_pred_ious, matched_gt_inds
    