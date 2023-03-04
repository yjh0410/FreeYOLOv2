import torch
from torchvision.ops import roi_align

import numpy as np
from pycocotools import mask as coco_mask


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        for i in range(len(polygons)):
            polygons[i] = polygons[i].reshape(-1).tolist()
        # for i in range(len(polygons)):
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2) # [H, W]
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, axis=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def refine_targets(bboxes, labels, masks, img_size, min_box_size, by_mask=False):
    """
        bboxes: (ndarray) [N, 4]
        labels: (ndarray) [N,]
        masks: (Tensor) [N, H, W]
        img_size: (Int) the long size of image
        min_box_size: (Int) the min size of target bbox
    """
    # refine target by the size of bbox
    if len(bboxes) > 0:
        # Cutout/Clip targets
        bboxes = torch.clamp(bboxes, 0, img_size)

        # check boxes
        bboxes_wh = bboxes[..., 2:] - bboxes[..., :2]
        min_tgt_boxes_size = torch.min(bboxes_wh, dim=-1)[0]

        keep = min_tgt_boxes_size.gt(min_box_size)

        bboxes = bboxes[keep]
        labels = labels[keep]
        masks = masks[keep]

    # refine target again by the area of mask
    if by_mask and len(masks) > 0:
        masks_sum = torch.sum(masks, dim=(1, 2)) # [N,]
        keep = masks_sum.gt(0)

        bboxes = bboxes[keep]
        labels = labels[keep]
        masks = masks[keep]

    # guard against no boxes via resizing
    bboxes = bboxes.view(-1, 4)
    labels = labels.view(-1)

    if masks.shape[0] > 0:
        img_h, img_w = masks.shape[1:]
        masks = masks.reshape(-1, img_h, img_w)
    else:
        img_h, img_w = masks.shape[1:]
        masks = torch.zeros([0, img_h, img_w], dtype=masks.dtype)

    return bboxes, labels, masks


def crop_and_resize(masks, boxes, mask_size):
    """
    Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
    This can be used to prepare training targets for Mask R-CNN.
    It has less reconstruction error compared to rasterization with polygons.
    However we observe no difference in accuracy,
    but BitMasks requires more memory to store all the masks.

    Args:
        boxes (Tensor): Nx4 tensor storing the boxes for each mask
        mask_size (int): the size of the rasterized mask.

    Returns:
        Tensor:
            A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
    """
    device = boxes.device

    batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
    rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

    bit_masks = masks.to(dtype=torch.float32)
    rois = rois.to(device=device, dtype=torch.float32)
    bit_masks = bit_masks[:, None, :, :]
    mask_sizes = (mask_size, mask_size)

    if bit_masks.is_quantized:
        bit_masks = input.dequantize()

    output = roi_align(bit_masks, rois, mask_sizes, 1.0, 0, True)
    output = output.squeeze(1)
    
    # threshold
    output = output >= 0.5

    return output
