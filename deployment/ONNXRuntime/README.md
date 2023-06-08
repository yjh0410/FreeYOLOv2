## FreeYOLOv2 ONNXRuntime

### Convert Your Model to ONNX

First, you should move to <FreeYOLO_HOME> by:
```shell
cd <FreeYOLO_HOME>
cd tools/
```
Then, you can:

1. Convert a standard FreeYOLO model by:
```shell
python3 export_onnx.py -m yolo_free_v2_nano --weight ../weight/coco/yolov1/yolo_free_v2_nano_coco.pth -nc 80 --img_size 640
```

Notes:
* -n: specify a model name. The model name must be one of the [yolox-s,m,l,x and yolox-nano, yolox-tiny, yolov3]
* -c: the model you have trained
* -o: opset version, default 11. **However, if you will further convert your onnx model to [OpenVINO](https://github.com/Megvii-BaseDetection/YOLOX/demo/OpenVINO/), please specify the opset version to 10.**
* --no-onnxsim: disable onnxsim
* To customize an input shape for onnx model,  modify the following code in tools/export_onnx.py:

    ```python
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size)
    ```

### ONNXRuntime Demo

Step1.
```shell
cd <YOLOX_HOME>/deployment/ONNXRuntime
```

Step2. 
```shell
python3 onnx_inference.py --model ../../weights/onnx/11/yolo_free_v2_nano.onnx -i ../test_image.jpg -s 0.3 --img_size 640
```
Notes:
* --model: your converted onnx model
* -i: input_image
* -s: score threshold for visualization.
* --img_size: should be consistent with the shape you used for onnx convertion.
