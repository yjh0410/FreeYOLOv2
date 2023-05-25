import json
import tempfile
import torch
import numpy as np
from dataset.ourdataset import OurDataset

try:
    from pycocotools.cocoeval import COCOeval
except:
    print("It seems that the COCOAPI is not installed.")


class OurDatasetEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, device, image_set='val', transform=None):
        """
        Args:
            data_dir (str): dataset root directory
            device: (int): CUDA or CPU.
            image_set: train or val.
            transform: used to preprocess inputs.
        """
        # ----------------- Basic parameters -----------------
        self.image_set = image_set
        self.transform = transform
        self.device = device
        # ----------------- Metrics -----------------
        self.map = 0.
        self.ap50_95 = 0.
        self.ap50 = 0.
        # ----------------- Dataset -----------------
        self.dataset = OurDataset(data_dir=data_dir, image_set=image_set)


    @torch.no_grad()
    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))

        # start testing
        for index in range(num_images): # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            # load an image
            img, id_ = self.dataset.pull_image(index)
            orig_h, orig_w, _ = img.shape

            # preprocess
            x, _, deltas = self.transform(img)
            x = x.unsqueeze(0).to(self.device) / 255.
            
            id_ = int(id_)
            ids.append(id_)
            # inference
            outputs = model(x)
            bboxes, scores, cls_inds = outputs

            # rescale
            img_h, img_w = x.shape[-2:]
            bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / (img_w - deltas[0]) * orig_w
            bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / (img_h - deltas[1]) * orig_h
            
            # clip bboxes
            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.class_ids[int(cls_inds[i])]
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i]) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score} # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, 'w'))
            cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
            print('ap50_95 : ', ap50_95)
            print('ap50 : ', ap50)
            self.map = ap50_95
            self.ap50_95 = ap50_95
            self.ap50 = ap50

            return ap50, ap50_95
        else:
            return 0, 0

