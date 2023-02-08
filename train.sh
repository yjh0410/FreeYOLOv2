# Train FreeYOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free_medium \
        -bs 16 \
        -accu 4 \
        -lr 0.01 \
        -mlr 0.01 \
        --max_epoch 300 \
        --wp_epoch 1 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        --resume weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
        # --pretrained weights/coco/yolo_free_medium/yolo_free_medium_39.46.pth \
