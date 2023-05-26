import os

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.crowdhuman_evaluator import CrowdHumanEvaluator
from evaluator.widerface_evaluator import WiderFaceEvaluator
from evaluator.ourdataset_evaluator import OurDatasetEvaluator



def build_evluator(args, data_cfg, transform, device):
    # Basic parameters
    data_dir = os.path.join(args.root, data_cfg['data_name'])

    # Evaluator
    ## VOC Evaluator
    if args.dataset == 'voc':
        evaluator = VOCAPIEvaluator(
            data_dir=data_dir,
            device=device,
            transform=transform)
    ## COCO Evaluator
    elif args.dataset == 'coco':
        evaluator = COCOAPIEvaluator(
            data_dir=data_dir,
            device=device,
            transform=transform
            )
    ## WiderFace Evaluator
    elif args.dataset == 'widerface':
        evaluator = WiderFaceEvaluator(
            data_dir=data_dir,
            device=device,
            image_set='val',
            transform=transform
        )
    ## CrowdHuman Evaluator
    elif args.dataset == 'crowdhuman':
        evaluator = CrowdHumanEvaluator(
            data_dir=data_dir,
            device=device,
            image_set='val',
            transform=transform
        )
    ## Custom dataset Evaluator
    elif args.dataset == 'ourdataset':
        evaluator = OurDatasetEvaluator(
            data_dir=data_dir,
            device=device,
            image_set='val',
            transform=transform
        )

    return evaluator
