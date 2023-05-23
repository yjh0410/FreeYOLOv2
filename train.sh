# Train FreeYOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -m yolo_free_v2_small \
        -bs 16 \
        -size 640 \
        --wp_epoch 3 \
        --max_epoch 300 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        --multi_scale \
        # --resume weights/coco/yolo_free_v2_large/yolo_free_v2_large_epoch_41_43.06.pth \
        # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
        # --eval_first
