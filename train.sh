# # Train FreeYOLO on COCO
# python train.py \
#         --cuda \
#         -d coco \
#         --root /mnt/share/ssd2/dataset/ \
#         -m yolo_free_v2_pico \
#         -bs 16 \
#         -size 640 \
#         --wp_epoch 3 \
#         --max_epoch 300 \
#         --eval_epoch 10 \
#         --ema \
#         --fp16 \
#         --multi_scale \
#         # --resume weights/coco/yolo_free_v2_large/yolo_free_v2_large_epoch_41_43.06.pth \
#         # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
#         # --eval_first

# Train FreeYOLO on WiderFace
python train.py \
        --cuda \
        -d widerface \
        --root /mnt/share/ssd2/dataset/ \
        -m yolo_free_v2_small \
        -bs 16 \
        -size 640 \
        --wp_epoch 1 \
        --max_epoch 100 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        --multi_scale \
        --mosaic 0.5 \
        --mixup 0.0 \
        --min_box_size 1 \
        --pretrained weights/coco/yolo_free_v2_small/yolo_free_v2_small_coco.pth \
        # --resume weights/coco/yolo_free_v2_large/yolo_free_v2_large_epoch_41_43.06.pth \
        # --eval_first

# # Train FreeYOLO on CrowdHuman
# python train.py \
#         --cuda \
#         -d crowdhuman \
#         --root /mnt/share/ssd2/dataset/ \
#         -m yolo_free_v2_large \
#         -bs 16 \
#         -size 640 \
#         --wp_epoch 1 \
#         --max_epoch 100 \
#         --eval_epoch 10 \
#         --ema \
#         --fp16 \
#         --multi_scale \
#         --pretrained weights/coco/yolo_free_v2_large/yolo_free_v2_large_coco.pth \
#         # --resume weights/coco/yolo_free_v2_large/yolo_free_v2_large_epoch_41_43.06.pth \
#         # --eval_first
