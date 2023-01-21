# Train FreeYOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free_nano \
        -bs 64 \
        -accu 1 \
        -lr 0.01 \
        -mlr 0.01 \
        --max_epoch 300 \
        --wp_epoch 3 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        # --pretrained weights/coco/yolo_free_large/yolo_free_large_47.1.pth \
        # --resume weights/coco/yolo_free_large/yolo_free_large_47.1.pth \
