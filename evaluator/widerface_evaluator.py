import json
import tempfile
import torch
from dataset.widerface import WiderFaceDataset

try:
    from pycocotools.cocoeval import COCOeval
except:
    print("It seems that the COCOAPI is not installed.")


class WiderFaceEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, device, image_set='val', transform=None):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataset = WiderFaceDataset(
            data_dir=data_dir,
            image_set=image_set,
            transform=None,
            mosaic_prob=0.,
            mixup_prob=0.
            )
        self.image_set = image_set
        self.transform = transform
        self.device = device

        self.map = 0.
        self.ap50_95 = 0.
        self.ap50 = 0.

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
            x = self.transform(img)[0]
            x = x.unsqueeze(0).to(self.device) / 255.
            
            id_ = int(id_)
            ids.append(id_)
            # inference
            outputs = model(x)
            bboxes, scores, cls_inds = outputs
            # rescale
            bboxes *= max(orig_h, orig_w)

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

