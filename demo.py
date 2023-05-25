import argparse
import cv2
import os
import time
import numpy as np
import imageio

import torch

# load transform
from dataset.transforms import build_transform

# load some utils
from utils.misc import load_weight
from utils.vis_tools import visualize

from config import build_config
from models.detectors import build_model



def parse_args():
    parser = argparse.ArgumentParser(description='FreeYOLOv2 Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='dataset/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='dataset/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/demos/',
                        type=str, help='The path to save the detection results')
    parser.add_argument('-vt', '--vis_thresh', default=0.2, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show visualization')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')

    # model
    parser.add_argument('-m', '--model', default='yolo_free_v2_nano', type=str,
                        help='build yolo_free_v2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.2, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.45, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument("--deploy", action="store_true", default=False,
                        help="deploy mode or not")
    parser.add_argument('--fuse_repconv', action='store_true', default=False,
                        help='fuse RepConv')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    return parser.parse_args()
                    

def detect(args,
           model, 
           device, 
           transform, 
           vis_thresh, 
           mode='image'):
    # class color
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    save_path = os.path.join(args.path_to_save, mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_video_name = os.path.join(save_path, cur_time+'.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
        print(save_video_name)
        image_list = []

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break
                orig_h, orig_w, _ = frame.shape

                # prepare
                x, _, deltas = transform(frame)
                x = x.unsqueeze(0).to(device) / 255.

                # inference
                t0 = time.time()
                bboxes, scores, labels = model(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale
                img_h, img_w = x.shape[-2:]
                bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / (img_w - deltas[0]) * orig_w
                bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / (img_h - deltas[1]) * orig_h

                # clip bbox
                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

                frame_vis = visualize(img=frame, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      vis_thresh=vis_thresh)
                frame_resized = cv2.resize(frame_vis, save_size)
                out.write(frame_resized)

                if args.gif:
                    gif_resized = cv2.resize(frame, (640, 480))
                    gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                    image_list.append(gif_resized_rgb)

                if args.show:
                    cv2.imshow('detection', frame_resized)
                    cv2.waitKey(1)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # generate GIF
        if args.gif:
            save_gif_path =  os.path.join(save_path, 'gif_files')
            os.makedirs(save_gif_path, exist_ok=True)
            save_gif_name = os.path.join(save_gif_path, '{}.gif'.format(cur_time))
            print('generating GIF ...')
            imageio.mimsave(save_gif_name, image_list, fps=fps)
            print('GIF done: {}'.format(save_gif_name))

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(args.path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_video_name = os.path.join(save_path, cur_time+'.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)
        print(save_video_name)
        image_list = []

        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                orig_h, orig_w, _ = frame.shape

                # prepare
                x, _, deltas = transform(frame)
                x = x.unsqueeze(0).to(device) / 255.

                # inference
                t0 = time.time()
                bboxes, scores, labels = model(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale
                img_h, img_w = x.shape[-2:]
                bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / (img_w - deltas[0]) * orig_w
                bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / (img_h - deltas[1]) * orig_h

                # clip bbox
                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

                # vis detection
                frame_vis = visualize(img=frame, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      vis_thresh=vis_thresh)

                frame_resized = cv2.resize(frame_vis, save_size)
                out.write(frame_resized)

                if args.gif:
                    gif_resized = cv2.resize(frame, (640, 480))
                    gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                    image_list.append(gif_resized_rgb)

                if args.show:
                    cv2.imshow('detection', frame_resized)
                    cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()

        # generate GIF
        if args.gif:
            save_gif_path =  os.path.join(save_path, 'gif_files')
            os.makedirs(save_gif_path, exist_ok=True)
            save_gif_name = os.path.join(save_gif_path, '{}.gif'.format(cur_time))
            print('generating GIF ...')
            imageio.mimsave(save_gif_name, image_list, fps=fps)
            print('GIF done: {}'.format(save_gif_name))

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for i, img_id in enumerate(os.listdir(args.path_to_img)):
            image = cv2.imread((args.path_to_img + '/' + img_id), cv2.IMREAD_COLOR)
            orig_h, orig_w, _ = image.shape

            # prepare
            x, _, deltas = transform(image)
            x = x.unsqueeze(0).to(device) / 255.

            t0 = time.time()
            # inference
            bboxes, scores, labels = model(x)
            print("detection time used ", time.time() - t0, "s")
            
            # rescale
            img_h, img_w = x.shape[-2:]
            bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / (img_w - deltas[0]) * orig_w
            bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / (img_h - deltas[1]) * orig_h

            # clip bbox
            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

            # vis detection
            img_vis = visualize(img=image, 
                                bboxes=bboxes,
                                scores=scores, 
                                labels=labels,
                                class_colors=class_colors,
                                vis_thresh=vis_thresh)
            cv2.imwrite(os.path.join(save_path, str(i).zfill(6)+'.jpg'), img_vis)
            if args.show:
                cv2.imshow('detection', img_vis)
                cv2.waitKey(0)


def run():
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(0)

    # config
    cfg = build_config(args)

    # build model
    model = build_model(args=args, 
                        cfg=cfg,
                        device=device, 
                        num_classes=80, 
                        trainable=False)

    # load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn, args.fuse_repconv)
    model.to(device).eval()

    # transform
    transform = build_transform(args.img_size, max_stride=max(cfg['stride']), is_train=False)

    print("================= DETECT =================")
    # run
    detect(args=args,
           model=model, 
            device=device,
            transform=transform,
            mode=args.mode,
            vis_thresh=args.vis_thresh)


if __name__ == '__main__':
    run()
