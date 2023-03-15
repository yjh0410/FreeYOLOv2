import torch
import torch.nn.functional as F
from utils.box_ops import *


class AlignSimOTA(object):
    def __init__(self, 
                 num_classes,
                 center_sampling_radius,
                 topk_candidate,
                 soft_center_radius = 3.0
                 ) -> None:
        self.num_classes = num_classes
        self.center_sampling_radius = center_sampling_radius
        self.topk_candidate = topk_candidate
        self.soft_center_radius = soft_center_radius


    @torch.no_grad()
    def __call__(self, 
                 fpn_strides, 
                 anchors, 
                 pred_cls, 
                 pred_box, 
                 tgt_labels,
                 tgt_bboxes):
        # [M,]
        strides = torch.cat([torch.ones_like(anchor_i[:, 0]) * stride_i
                                for stride_i, anchor_i in zip(fpn_strides, anchors)], dim=-1)
        # List[F, M, 2] -> [M, 2]
        anchors = torch.cat(anchors, dim=0)
        num_anchor = anchors.shape[0]        
        num_gt = len(tgt_labels)

        fg_mask, is_in_boxes_and_center = \
            self.get_in_boxes_info(
                tgt_bboxes,
                anchors,
                strides,
                num_anchor,
                num_gt
                )

        cls_preds_ = pred_cls[fg_mask]   # [Mp, C]
        box_preds_ = pred_box[fg_mask]   # [Mp, 4]
        num_in_boxes_anchor = box_preds_.shape[0]

        # ---------------------------- ctr cost ----------------------------
        gt_center = self.get_box_center(tgt_bboxes)
        anchors_fg = anchors[fg_mask]
        strides_fg = strides[fg_mask]
        distance = (anchors_fg[None] - gt_center[:, None, :]).pow(2).sum(-1).sqrt() / strides_fg[None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        # ---------------------------- reg cost ----------------------------
        # iou [N, Mp]
        pair_wise_ious, _ = box_iou(tgt_bboxes, box_preds_)
        # ioui cost [N, Mp]
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # ---------------------------- cls cost ----------------------------
        # [N, C] -> [N, Mp, C]
        gt_cls = (
            F.one_hot(tgt_labels.long(), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        # iou-aware gt cls label
        soft_gt_cls = gt_cls * pair_wise_ious.unsqueeze(-1)

        with torch.cuda.amp.autocast(enabled=False):
            # [N, Mp, C]
            score_preds = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            scale_factor = score_preds - soft_gt_cls
            # cls cost
            pair_wise_cls_loss = F.binary_cross_entropy(
                score_preds, soft_gt_cls, reduction="none"
            ) * scale_factor.abs().pow(2.0)
            pair_wise_cls_loss = pair_wise_cls_loss.sum(-1) # [N, Mp]
        del score_preds

        cost = (
            pair_wise_cls_loss
            + pair_wise_ious_loss
            + soft_center_prior
            + 100000.0 * (~is_in_boxes_and_center)
        ) # [N, Mp]

        (
            num_fg,
            gt_matched_classes,         # [num_fg,]
            pred_ious_this_matching,    # [num_fg,]
            matched_gt_inds,            # [num_fg,]
        ) = self.dynamic_k_matching(
            cost,
            pair_wise_ious,
            tgt_labels,
            num_gt,
            fg_mask
            )
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
                gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg,
        )


    def get_box_center(self, boxes, box_dim: int = 4):
        """Return a tensor representing the centers of boxes.

        Args:
            boxes (Tensor): Boxes tensor. Has shape of (b, n, box_dim)
            box_dim (int): The dimension of box. 4 means horizontal box and
                5 means rotated box. Defaults to 4.

        Returns:
            Tensor: Centers have shape of (b, n, 2)
        """
        if box_dim == 4:
            # Horizontal Boxes, (x1, y1, x2, y2)
            return (boxes[..., :2] + boxes[..., 2:]) / 2.0
        elif box_dim == 5:
            # Rotated Boxes, (x, y, w, h, a)
            return boxes[..., :2]
        else:
            raise NotImplementedError(f'Unsupported box_dim:{box_dim}')


    def get_in_boxes_info(
        self,
        gt_bboxes,   # [N, 4]
        anchors,     # [M, 2]
        strides,     # [M,]
        num_anchors, # M
        num_gt,      # N
        ):
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
        # in fixed center
        center_radius = self.center_sampling_radius

        # [N, 2]
        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5
        
        # [1, M]
        center_radius_ = center_radius * strides.unsqueeze(0)

        gt_bboxes_l = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) - center_radius_ # x1
        gt_bboxes_t = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) - center_radius_ # y1
        gt_bboxes_r = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) + center_radius_ # x2
        gt_bboxes_b = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) + center_radius_ # y2

        c_l = x_centers - gt_bboxes_l
        c_r = gt_bboxes_r - x_centers
        c_t = y_centers - gt_bboxes_t
        c_b = gt_bboxes_b - y_centers
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center
    
    
    def dynamic_k_matching(
        self, 
        cost, 
        pair_wise_ious, 
        gt_classes, 
        num_gt, 
        fg_mask
        ):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.topk_candidate, ious_in_boxes_matrix.size(1))
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

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds