import os
import cv2
import random
import numpy as np

from torch.utils.data import Dataset

try:
    from pycocotools.coco import COCO
except:
    print("It seems that the COCOAPI is not installed.")

try:
    from .transforms import yolov5_mosaic_augment, yolov5_mosaic_augment_9x, yolov5_mixup_augment
except:
    from transforms import yolov5_mosaic_augment, yolov5_mosaic_augment_9x, yolov5_mixup_augment


crowd_class_labels = ('person',)



class CrowdHumanDataset(Dataset):
    """
    CrowdHuman dataset class.
    """
    def __init__(self, 
                 img_size=640,
                 data_dir=None, 
                 image_set='train',
                 transform=None,
                 trans_config=None,
                 is_train=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.img_size = img_size
        self.image_set = image_set
        self.json_file = '{}.json'.format(image_set)
        self.data_dir = data_dir
        self.coco = COCO(os.path.join(self.data_dir, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.is_train = is_train

        # augmentation
        self.transform = transform
        self.mosaic_prob = 0
        self.mosaic_9x_prob = 0
        self.mixup_prob = 0
        if trans_config is not None:
            self.mosaic_prob = trans_config['mosaic_prob']
            self.mosaic_9x_prob = trans_config['mosaic_9x_prob']
            self.mixup_prob = trans_config['mixup_prob']
            self.trans_config = trans_config

        print('==============================')
        print('Image Set: {}'.format(image_set))
        print('Json file: {}'.format(self.json_file))
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        return self.pull_item(index)


    def load_image_target(self, index):
        # load an image
        image, _ = self.pull_image(index)
        height, width, channels = image.shape

        # load a target
        bboxes, labels = self.pull_anno(index)

        target = {
            "boxes": bboxes,
            "labels": labels,
            "orig_size": [height, width]
        }

        return image, target


    def load_mosaic(self, index):
        if random.random() > self.mosaic_9x_prob:
            load_mosaic_4x = True
            # load 4x mosaic image
            index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
            id1 = index
            id2, id3, id4 = random.sample(index_list, 3)
            indexs = [id1, id2, id3, id4]
        else:
            load_mosaic_4x = False
            # load 9x mosaic image
            index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
            id1 = index
            id2_9 = random.sample(index_list, 8)
            indexs = [id1] + id2_9

        # load images and targets
        image_list = []
        target_list = []
        for index in indexs:
            img_i, target_i = self.load_image_target(index)
            image_list.append(img_i)
            target_list.append(target_i)

        # Mosaic Augment
        if load_mosaic_4x:
            image, target = yolov5_mosaic_augment(
                image_list, target_list, self.img_size, self.trans_config)
        else:
            image, target = yolov5_mosaic_augment_9x(
                image_list, target_list, self.img_size, self.trans_config)
                
        return image, target

        
    def load_mixup(self, origin_image, origin_target):
        new_index = np.random.randint(0, len(self.ids))
        new_image, new_target = self.load_mosaic(new_index)
        image, target = yolov5_mixup_augment(
            origin_image, origin_target, new_image, new_target)

        return image, target
    

    def pull_item(self, index):
        if random.random() < self.mosaic_prob:
            # load a mosaic image
            mosaic = True
            image, target = self.load_mosaic(index)
        else:
            mosaic = False
            # load an image and target
            image, target = self.load_image_target(index)

        # MixUp
        if random.random() < self.mixup_prob:
            image, target = self.load_mixup(image, target)

        # augment
        image, target, deltas = self.transform(image, target, mosaic)

        return image, target, deltas


    def pull_image(self, index):
        id_ = self.ids[index]
        im_ann = self.coco.loadImgs(id_)[0] 
        img_file = os.path.join(
                self.data_dir, 'CrowdHuman_{}'.format(self.image_set), 'Images', im_ann["file_name"])
        image = cv2.imread(img_file)

        return image, id_


    def pull_anno(self, index):
        img_id = self.ids[index]
        im_ann = self.coco.loadImgs(img_id)[0]
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=0)
        annotations = self.coco.loadAnns(anno_ids)
        
        # image infor
        width = im_ann['width']
        height = im_ann['height']
        
        #load a target
        bboxes = []
        labels = []
        for anno in annotations:
            if 'bbox' in anno and anno['area'] > 0:
                # bbox
                x1 = np.max((0, anno['bbox'][0]))
                y1 = np.max((0, anno['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
                if x2 <= x1 or y2 <= y1:
                    continue
                # class label
                cls_id = self.class_ids.index(anno['category_id'])
                
                bboxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

        # guard against no boxes via resizing
        bboxes = np.array(bboxes).reshape(-1, 4)
        labels = np.array(labels).reshape(-1)
        
        return bboxes, labels


if __name__ == "__main__":
    import argparse
    from transforms import build_transform
    
    parser = argparse.ArgumentParser(description='FreeYOLO')

    # opt
    parser.add_argument('--root', default='D:\\python_work\\object-detection\\dataset\\CrowdHuman',
                        help='data root')

    args = parser.parse_args()
    
    img_size = 640
    is_train = False
    trans_config = {
        # Basic Augment
        'degrees': 0.0,
        'translate': 0.2,
        'scale': 0.9,
        'shear': 0.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        # Mosaic & Mixup
        'mosaic_prob': 0.0,
        'mosaic_9x_prob': 0.2,
        'mixup_prob': 0.0,
        'mosaic_type': 'yolov5_mosaic',
        'mixup_type': 'yolov5_mixup',
        'mixup_scale': [0.5, 1.5]
    }
    transform = build_transform(img_size, trans_config, max_stride=32, is_train=is_train)

    dataset = CrowdHumanDataset(
        img_size=img_size,
        data_dir=args.root,
        image_set='val',
        transform=transform,
        trans_config=trans_config,
        )
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target, _ = dataset.pull_item(i)
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cls_id = int(label)
            color = class_colors[cls_id]
            # class name
            label = crowd_class_labels[cls_id]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)