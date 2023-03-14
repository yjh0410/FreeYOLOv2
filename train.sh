# Train FreeYOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free_v2_large \
        -bs 16 \
        -size 640 \
        --wp_epoch 3 \
        --max_epoch 300 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        --multi_scale \
        # --resume weights/coco/yolo_free_v2_nano/yolo_free_v2_nano_epoch_31_21.34.pth \
        # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \


# # Debug FreeYOLO on VOC
# python train.py \
#         --cuda \
#         -d voc \
#         --root /mnt/share/ssd2/dataset/ \
#         -v yolo_free_v2_tiny \
#         -bs 16 \
#         --max_epoch 25 \
#         --wp_epoch 1 \
#         --eval_epoch 5 \
#         --ema \
#         --fp16 \
#         # --resume weights/coco/yolo_free_medium/yolo_free_medium_epoch_31_39.46.pth \
#         # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
