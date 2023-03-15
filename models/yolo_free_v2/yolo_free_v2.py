import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo_free_v2_backbone import build_backbone
from .yolo_free_v2_neck import build_neck
from .yolo_free_v2_pafpn import build_fpn
from .yolo_free_v2_head import build_head

from utils.nms import non_max_suppression


# Anchor-free YOLO
class FreeYOLOv2(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 max_det = 1000,
                 no_decode = False):
        super(FreeYOLOv2, self).__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.reg_max = cfg['reg_max']
        self.use_dfl = cfg['reg_max'] > 1
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_det = max_det
        self.no_decode = no_decode
        
        # --------- Network Parameters ----------
        self.proj_conv = nn.Conv2d(self.reg_max, 1, kernel_size=1, bias=False)

        ## backbone
        self.backbone, feats_dim = build_backbone(cfg=cfg)

        ## neck
        self.neck = build_neck(cfg=cfg, in_dim=feats_dim[-1], out_dim=feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim
        
        ## fpn
        self.fpn = build_fpn(cfg=cfg, in_dims=feats_dim)
        fpn_dims = self.fpn.out_dim

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg, feat_dim, fpn_dims, num_classes) 
            for feat_dim in fpn_dims
            ])

        ## pred
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head.cls_out_dim, self.num_classes, kernel_size=1) 
                                for head in self.non_shared_heads
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_out_dim, 4*(cfg['reg_max']), kernel_size=1) 
                                for head in self.non_shared_heads
                              ])                 

        # --------- Network Initialization ----------
        # init bias
        self.init_yolo()


    def init_yolo(self): 
        # Init yolo
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03    
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # cls pred
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for reg_pred in self.reg_preds:
            b = reg_pred.bias.view(-1, )
            b.data.fill_(1.0)
            reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = reg_pred.weight
            w.data.fill_(0.)
            reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max, 1, 1]).clone().detach(),
                                                   requires_grad=False)


    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)

        return anchors
        

    def decode_boxes(self, anchors, pred_regs, stride):
        """
        Input:
            anchors:  (List[Tensor]) [1, M, 2]
            pred_reg: (List[Tensor]) [B, M, 4*(reg_max)]
        Output:
            pred_box: (Tensor) [B, M, 4]
        """
        # tlbr -> xyxy
        pred_x1y1 = anchors - pred_regs[..., :2] * stride
        pred_x2y2 = anchors + pred_regs[..., 2:] * stride
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    @torch.no_grad()
    def inference_single_image(self, x):
        # backbone
        pyramid_feats = self.backbone(x)

        # neck
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # fpn
        pyramid_feats = self.fpn(pyramid_feats)

        # non-shared heads
        all_preds = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # pred
            cls_pred = self.cls_preds[level](cls_feat)  # [B, C, H, W]
            reg_pred = self.reg_preds[level](reg_feat)  # [B, 4*(reg_max), H, W]

            if self.no_decode:
                anchors = None
            else:
                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # [M, 4]
                anchors = self.generate_anchors(level, fmp_size)

            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

            if self.use_dfl:
                B, M = cls_pred.shape[:2]
                # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max] -> [B, 4, M, reg_max]
                reg_pred = reg_pred.reshape([B, M, 4, self.reg_max])
                # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
                reg_pred = reg_pred.permute(0, 3, 2, 1).contiguous()
                # [B, reg_max, 4, M] -> [B, 1, 4, M]
                reg_pred = self.proj_conv(F.softmax(reg_pred, dim=1))
                # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
                reg_pred = reg_pred.view(B, 4, M).permute(0, 2, 1).contiguous()

            # decode box
            box_pred = self.decode_boxes(anchors, reg_pred, stride=self.stride[level])

            # [B, M, 4 + C]
            preds = torch.cat([cls_pred.sigmoid(), box_pred], dim=-1)
            all_preds.append(preds)

        # NMS
        preds = torch.cat(all_preds, dim=1)
        outputs = non_max_suppression(
            preds, self.conf_thresh, self.nms_thresh, classes=None, agnostic=False, max_det=self.max_det)
        
        # Batch size = 1
        bboxes = outputs[0][..., :4].float().cpu().numpy()
        scores = outputs[0][..., 4].float().cpu().numpy()
        labels = outputs[0][..., 5].long().cpu().numpy()
        
        return bboxes, scores, labels


    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            pyramid_feats = self.backbone(x)

            # neck
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])

            # fpn
            pyramid_feats = self.fpn(pyramid_feats)

            # non-shared heads
            all_anchors = []
            all_cls_preds = []
            all_reg_preds = []
            all_box_preds = []
            all_strides = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
                cls_feat, reg_feat = head(feat)

                # pred
                cls_pred = self.cls_preds[level](cls_feat)  # [B, C, H, W]
                reg_pred = self.reg_preds[level](reg_feat)  # [B, 4*(reg_max), H, W]

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # generate anchor boxes: [M, 2]
                anchors = self.generate_anchors(level, fmp_size)
                
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

                # decode box: [B, M, 4]
                box_pred = self.decode_boxes(anchors, reg_pred, self.stride[level])

                # stride tensor: [M, 1]
                stride_tensor = torch.ones_like(anchors[..., :1]) * self.stride[level]

                all_cls_preds.append(cls_pred)
                all_reg_preds.append(reg_pred)
                all_box_preds.append(box_pred)
                all_anchors.append(anchors)
                all_strides.append(stride_tensor)
            
            # output dict
            outputs = {"pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                       "pred_reg": all_reg_preds,        # List(Tensor) [B, M, 4*(reg_max)]
                       "pred_box": all_box_preds,        # List(Tensor) [B, M, 4]
                       "anchors": all_anchors,           # List(Tensor) [M, 2]
                       "strides": self.stride,           # List(Int) = [8, 16, 32]
                       "stride_tensor": all_strides      # List(Tensor) [M, 1]
                       }           
            return outputs