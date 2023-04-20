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

        # check gt
        if num_gt == 0 or gt_bboxes.max().item() == 0.:
            return {
                'assigned_labels':
                gt_labels.new_full(
                    pred_cls[..., 0].shape,
                    self.num_classes,
                    dtype=torch.long),
                'assigned_bboxes':
                gt_bboxes.new_full(pred_cls.shape, 0),
                'assign_metrics':
                gt_bboxes.new_full(pred_cls[..., 0].shape, 0)
            }
        
        # get inside points: [N, M]
        is_in_gt = self.find_inside_points(gt_bboxes, anchors)
        valid_mask = is_in_gt.sum(dim=0) > 0

        num_anchors = pred_box.shape[0]

        # ----------------------------------- soft center prior -----------------------------------
        gt_center = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2.0
        distance = (anchors.unsqueeze(0) - gt_center.unsqueeze(1)
                    ).pow(2).sum(-1).sqrt() / strides.unsqueeze(0)  # [N, M]
        distance = distance * valid_mask.unsqueeze(0)
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        # ----------------------------------- regression cost -----------------------------------
        pair_wise_ious, _ = box_iou(gt_bboxes, pred_box)  # [N, M]
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8) * self.iou_weight

        # ----------------------------------- classification cost -----------------------------------
        gt_cls = (
            F.one_hot(gt_labels.long(), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_anchors, 1)
        ) # [N, C] -> [N, M, C]
        soften_gt_cls = gt_cls * pair_wise_ious.unsqueeze(-1)
        with torch.cuda.amp.autocast(enabled=False):
            # [M, C] -> [N, M, C]
            pairwise_pred_scores = pred_cls.float().unsqueeze(0).repeat(num_gt, 1, 1) # [N, M, C]
            scale_factor = (soften_gt_cls - pairwise_pred_scores.sigmoid()).abs().pow(2.0)
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                pairwise_pred_scores, soften_gt_cls,
                reduction="none") * scale_factor
            pair_wise_cls_loss = pair_wise_cls_loss.sum(-1) # [N, M]
            
        del pairwise_pred_scores

        # foreground cost matrix
        cost_matrix = pair_wise_cls_loss + pair_wise_ious_loss + soft_center_prior
        max_pad_value = torch.ones_like(cost_matrix) * 1e9
        cost_matrix = torch.where(valid_mask[None].repeat(num_gt, 1),
                                  cost_matrix, max_pad_value)

        # dynamic label assignment
        (
            matched_pred_ious,
            matched_gt_inds,
            fg_mask_inboxes
        ) = self.dynamic_k_matching(
            cost_matrix,
            pair_wise_ious,
            num_gt
            )
        del pair_wise_cls_loss, cost_matrix, pair_wise_ious, pair_wise_ious_loss

        # process assigned labels
        assigned_labels = gt_labels.new_full(pred_cls[..., 0].shape,
                                             self.num_classes)  # [M,]
        assigned_labels[fg_mask_inboxes] = gt_labels[matched_gt_inds].squeeze(-1)
        assigned_labels = assigned_labels.long()  # [M,]

        assigned_bboxes = gt_bboxes.new_full(pred_box.shape, 0)        # [M, 4]
        assigned_bboxes[fg_mask_inboxes] = gt_bboxes[matched_gt_inds]  # [M, 4]

        assign_metrics = gt_bboxes.new_full(pred_cls[..., 0].shape, 0) # [M, 4]
        assign_metrics[fg_mask_inboxes] = matched_pred_ious            # [M, 4]

        assigned_dict = dict(
            assigned_labels=assigned_labels,
            assigned_bboxes=assigned_bboxes,
            assign_metrics=assign_metrics
            )
        
        return assigned_dict


    def find_inside_points(self, gt_bboxes, anchors):
        """
            gt_bboxes: Tensor -> [N, 2]
            anchors:   Tensor -> [M, 2]
        """
        num_anchors = anchors.shape[0]
        num_gt = gt_bboxes.shape[0]

        anchors_expand = anchors.unsqueeze(0).repeat(num_gt, 1, 1)           # [N, M, 2]
        gt_bboxes_expand = gt_bboxes.unsqueeze(1).repeat(1, num_anchors, 1)  # [N, M, 4]

        # offset
        lt = anchors_expand - gt_bboxes_expand[..., :2]
        rb = gt_bboxes_expand[..., 2:] - anchors_expand


        bbox_deltas = torch.cat([lt, rb], dim=-1)

        is_in_gts = bbox_deltas.min(dim=-1).values > 0

        return is_in_gts
    

    def dynamic_k_matching(self, cost_matrix, pairwise_ious, num_gt):
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets.

        Args:
            cost_matrix (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        Returns:
            tuple: matched ious and gt indexes.
        """
        matching_matrix = torch.zeros_like(cost_matrix, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(1))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=1)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        # sorting the batch cost matirx is faster than topk
        _, sorted_indices = torch.sort(cost_matrix, dim=1)
        for gt_idx in range(num_gt):
            topk_ids = sorted_indices[gt_idx, :dynamic_ks[gt_idx]]
            matching_matrix[gt_idx, :][topk_ids] = 1

        del topk_ious, dynamic_ks

        prior_match_gt_mask = matching_matrix.sum(0) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost_matrix[:, prior_match_gt_mask], dim=0)
            matching_matrix[:, prior_match_gt_mask] *= 0
            matching_matrix[cost_argmin, prior_match_gt_mask] = 1

        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(0)[fg_mask_inboxes]
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        return matched_pred_ious, matched_gt_inds, fg_mask_inboxes


    def _dynamic_k_matching(
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

        is_in_gt[is_in_gt.clone()] = fg_mask_inboxes
        fg_mask = is_in_gt

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        assigned_labels = gt_classes[matched_gt_inds]

        matched_pred_ious = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return fg_mask, assigned_labels, matched_pred_ious, matched_gt_inds
    