import cv2
import numpy as np
from dataset.coco import coco_class_index, coco_class_labels


def plot_bbox_labels(img, bbox, label, cls_color, test_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    # plot title bbox
    cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * test_scale), y1), cls_color, -1)
    # put the test on the title bbox
    cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, test_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, bboxes, scores, labels, class_colors, vis_thresh=0.3):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_color = class_colors[int(labels[i])]
            cls_id = coco_class_index[int(labels[i])]
            mess = '%s: %.2f' % (coco_class_labels[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, test_scale=ts)

    return img


def vis_data(images, targets):
    """
        images: (tensor) [B, 3, H, W]
        targets: (list) a list of targets
    """
    batch_size = images.size(0)
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        target = targets[bi]
        image = image.astype(np.uint8)
        image = image.copy()

        tgt_boxes = target['boxes']
        tgt_labels = target['labels']
        for box, label in zip(tgt_boxes, tgt_labels):
            x1, y1, x2, y2 = box
            cls_id = int(label)
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            color = class_colors[cls_id]
            # draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        cv2.imshow('train target', image)
        cv2.waitKey(0)
