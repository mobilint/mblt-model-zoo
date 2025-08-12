# Pre-Trained Vision Models
Here, we give the full list of publicly pre-trained models supported by the Mobilint Model Zoo.
## Image Classification (ImageNet)

| Model | Input Size <br> (H, W, C)|Top1 Acc <br> (NPU)| Top1 Acc <br> (GPU)| FLOPs (B) | params (M) |Source|Note|
|------------|------------|-----------|--------------------|--------|-------|------|------|
| AlexNet | (224,224,3)| 56.022 | 56.552 | 0.71 | 61.10 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html) |  |
| ConvNeXt_Tiny | (224,224,3) | 82.302 | 82.460| 4.46 | 28.59 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html) |  |
| ConvNeXt_Small | (224,224,3) | 83.432 | 83.560 | 8.68 | 50.22 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_small.html) |  |
| ConvNeXt_Base | (224,224,3) | 83.834 | 84.050 | 15.36 | 88.59 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html) |  |
| ConvNeXt_Large | (224,224,3) | 83.276 | 84.410 | 34.36 | 197.77 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_large.html) |  |
| DenseNet121 | (224,224,3) | 74.194 | 74.414 | 2.83 | 7.98 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html) |  |
| DenseNet169 | (224,224,3) | 75.476 | 75.566 | 3.36 | 14.15 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet169.html) |  |
| DenseNet201 | (224,224,3) | 76.650 | 76.880 | 4.29 | 20.01 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet201.html) |  |
| GoogLeNet | (224,224,3) | 69.566 | 69.780 | 1.50 | 6.62 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.googlenet.html)	|  |
| Inception_V3 | (299,299,3) | 77.120 | 77.278 | 5.71 | 27.16 | [Link](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html)|  |
| MNASNet1_0 | (224,224,3)| 72.696 | 73.422 | 0.31 | 4.38 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mnasnet1_0.html) | |
| MNASNet1_3 | (224,224,3) | 75.720 | 76.466 | 0.53 | 6.28 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mnasnet1_3.html) | |
| MobileNet_V2 | (224,224,3)| 71.624 | 72.138 | 0.30 | 3.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mobilenet_v2.html) | IMAGENET1K_V2 |
| RegNet_X_400MF | (224,224,3) | 72.544 | 72.900 | 0.41 | 5.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_400mf.html) | IMAGENET1K_V1 |
| RegNet_X_400MF | (224,224,3) | 74.218 | 74.846 | 0.41 | 5.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_400mf.html) | IMAGENET1K_V2 |
| RegNet_X_800MF | (224,224,3) | 74.928 | 75.204 | 0.80 | 7.26 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_800mf.html) | IMAGENET1K_V1 |
| RegNet_X_800MF | (224,224,3) | 76.984 | 77.488 | 0.80 | 7.26 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_800mf.html) | IMAGENET1K_V2 |
| RegNet_X_1_6GF | (224,224,3) | 76.756 | 77.080 | 1.60 | 9.19 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_1_6gf.html) | IMAGENET1K_V1 |
| RegNet_X_1_6GF | (224,224,3) | 79.168 | 79.670 | 1.60 | 9.19 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_1_6gf.html) | IMAGENET1K_V2 |
| RegNet_X_3_2GF | (224,224,3) | 77.978 | 78.346 | 3.18 | 15.30 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html) | IMAGENET1K_V1 |
| RegNet_X_3_2GF | (224,224,3) | 80.792 | 81.188 | 3.18 | 15.30 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html) | IMAGENET1K_V2 |
| RegNet_X_8GF | (224,224,3) | 79.322 | 79.368 | 8.00 | 39.57 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_8gf.html) | IMAGENET1K_V1 |
| RegNet_X_8GF | (224,224,3) | 81.388 | 81.680 | 8.00 | 39.57 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_8gf.html) | IMAGENET1K_V2 |
| RegNet_X_16GF | (224,224,3) | 79.884 | 80.090 | 15.94 | 54.28 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_16gf.html) | IMAGENET1K_V1 |
| RegNet_X_16GF | (224,224,3) | 82.312 | 82.712 | 15.94 | 54.28 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_16gf.html) | IMAGENET1K_V2 |
| RegNet_X_32GF | (224,224,3) | 80.482 | 80.592 | 31.74 | 107.81 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_32gf.html) | IMAGENET1K_V1 |
| RegNet_X_32GF | (224,224,3) | 82.722 | 83.014 | 31.74 | 107.81 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_32gf.html) | IMAGENET1K_V2 |
| RegNet_Y_400MF | (224,224,3) | 73.594 | 73.998 | 0.40 | 4.34 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html) | IMAGENET1K_V1 |
| RegNet_Y_400MF | (224,224,3) | 75.294 | 75.804 | 0.40 | 4.34 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html) | IMAGENET1K_V2 |
| RegNet_Y_800MF | (224,224,3) | 76.132 | 76.406 | 0.83 | 6.43 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_800mf.html) | IMAGENET1K_V1|
| RegNet_Y_800MF | (224,224,3) | 78.280 | 78.904 | 0.83 | 6.43 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_800mf.html) | IMAGENET1K_V2|
| RegNet_Y_1_6GF | (224,224,3) | 77.260 | 77.934 | 1.61 | 11.20 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_1_6gf.html) | IMAGENET1K_V1 |
| RegNet_Y_1_6GF | (224,224,3) | 80.298 | 80.876 | 1.61 | 11.20 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_1_6gf.html) | IMAGENET1K_V2 |
| RegNet_Y_3_2GF | (224,224,3) | 78.504 | 78.962 | 3.18 | 19.44 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_y_3_2gf.html) | IMAGENET1K_V1 |
| RegNet_Y_3_2GF | (224,224,3) | 81.354 | 82.010 | 3.18 | 19.44 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_y_3_2gf.html) | IMAGENET1K_V2 |
| RegNet_Y_8GF | (224,224,3) | 79.768 | 80.052 | 8.47 | 39.38 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_8gf.html) | IMAGENET1K_V1 |
| RegNet_Y_8GF | (224,224,3) | 82.418 | 82.822 | 8.47 | 39.38 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_8gf.html) | IMAGENET1K_V2 |
| RegNet_Y_16GF | (224,224,3) | 80.240 | 80.428 | 15.91 | 83.59 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_16gf.html)| IMAGENET1K_V1 |
| RegNet_Y_16GF | (224,224,3) | 82.348 | 82.868 | 15.91 | 83.59 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_16gf.html)| IMAGENET1K_V2 |
| RegNet_Y_32GF | (224,224,3) | 80.510 | 80.840 | 32.28 | 145.05 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_32gf.html) | IMAGENET1K_V1 |
| RegNet_Y_32GF | (224,224,3) | 82.798 | 83.362 | 32.28 | 145.05 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_32gf.html) | IMAGENET1K_V2 |
| ResNet18 | (224,224,3) | 69.494 | 69.774 | 1.81 | 11.69 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet18.html) | |
| ResNet34 | (224,224,3) | 73.134 | 73.310 | 3.66 | 21.80 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet34.html) | |
| ResNet50 | (224,224,3) | 75.946 | 76.128 | 4.09 | 25.56 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet50.html) | IMAGENET1K_V1 |
| ResNet50 | (224,224,3) | 80.326 | 80.844 | 4.09 | 25.56 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet50.html) | IMAGENET1K_V2 |
| ResNet101 | (224,224,3) | 77.112 | 77.362 | 7.80 | 44.55 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet101.html) | IMAGENET1K_V1 |
| ResNet101 | (224,224,3) | 81.472 | 81.918 | 7.80 | 44.55 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet101.html) | IMAGENET1K_V2 |
| ResNet152 | (224,224,3) | 77.922 | 78.316 | 11.51 | 60.19 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet152.html) | IMAGENET1K_V1 |
| ResNet152 | (224,224,3) | 81.950 | 82.272 | 11.51 | 60.19 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet152.html) | IMAGENET1K_V2 |
| ResNeXt50_32X4D | (224,224,3) | 77.528 | 77.630 | 4.23 | 25.03 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext50_32x4d.html) | IMAGENET1K_V1 |
| ResNeXt50_32X4D | (224,224,3) | 80.728 | 81.220 | 4.23 | 25.03 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext50_32x4d.html) | IMAGENET1K_V2 |
| ResNeXt101_32X8D | (224,224,3) | 78.998 | 79.288 | 16.41 | 88.79 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_32x8d.html) | IMAGENET1K_V1 |
| ResNeXt101_32X8D | (224,224,3) | 82.510 | 82.780 | 16.41 | 88.79 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_32x8d.html) | IMAGENET1K_V2 |
| ResNeXt101_64X4D | (224,224,3) | 82.876 | 83.236 | 15.46 | 83.46 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_64x4d.html) | |
| ShuffleNet_V2_X1_0 | (224,224,3) | 68.616 | 69.312 | 0.14 | 2.28 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_0.html) | |
| ShuffleNet_V2_X1_5 | (224,224,3) | 72.240 | 72.966 | 0.30 | 3.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_5.html) | |
| ShuffleNet_V2_X2_0 | (224,224,3) | 75.540 | 76.222 | 0.58 | 7.39 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x2_0.html) | |
| VGG11 | (224,224,3) | 68.616 | 68.978 | 7.61 | 132.86 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg11.html) | |
| VGG11_BN | (224,224,3) | 70.010 | 70.328 | 7.61 | 132.87 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg11_bn.html) | |
| VGG13 | (224,224,3) | 69.588 | 69.886 | 11.31 | 133.05 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg13.html) | |
| VGG13_BN | (224,224,3) | 71.262 | 71.570 | 11.31 | 133.05 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg13_bn.html) | |
| VGG16 | (224,224,3) | 71.350 | 71.610 | 15.47 | 138.36 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg16.html) | |
| VGG16_BN | (224,224,3) | 73.178 | 73.408 | 15.47 | 138.37 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg16_bn.html) | |
| VGG19 | (224,224,3) | 72.140 | 72.384 | 19.63 | 143.67 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg19.html) | |
| VGG19_BN | (224,224,3) | 73.898 | 74.166 | 19.63 | 143.68 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg19_bn.html) | |
| Wide_ResNet50_2 | (224,224,3) | 78.268 | 78.490 | 11.40 | 68.88 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html) | IMAGENET1K_V1 |
| Wide_ResNet50_2 | (224,224,3) | 81.226 | 81.626 | 11.40 | 68.88 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html) | IMAGENET1K_V2 |
| Wide_ResNet101_2 | (224,224,3) | 78.384 | 78.834 | 22.75 | 126.89 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet101_2.html) | IMAGENET1K_V1 |
| Wide_ResNet101_2 | (224,224,3) | 82.282 | 82.504 | 22.75 | 126.89 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet101_2.html) | IMAGENET1K_V2 |

## Object Detection (COCO)

| Model | Input Size <br> (H, W, C)| mAP <br> (NPU) | mAP <br> (GPU)| FLOPs (B) | params (M) | Source | Note |
|-------|--------------------------|----------------|---------------|-----------|------------|--------|------|
| yolov3u | (640,640,3) | 51.053 | 51.577 | 282.68  | 103.73 | [Link](https://docs.ultralytics.com/models/yolov3/)| |
| yolov3-spp | (640,640,3) | 46.715 | 47.613 | 156.91 | 62.97 | [Link](https://github.com/ultralytics/yolov3/)| |
| yolov3-sppu | (640,640,3) | 51.382 | 51.763 | 283.57 | 104.78 | [Link](https://docs.ultralytics.com/models/yolov3/)| |
| yolov5su | (640, 640, 3) | 42.206 | 42.871 | 24.03 | 9.14 | [Link](https://docs.ultralytics.com/models/yolov5/)| |
| yolov5mu | (640, 640, 3) | 48.217 | 48.906 | 64.27 | 25.09 | [Link](https://docs.ultralytics.com/models/yolov5/)| |
| yolov5lu | (640, 640, 3) | 51.718 | 52.171 | 135.09 | 53.19 | [Link](https://docs.ultralytics.com/models/yolov5/)| |
| yolov5l6 | (1280, 1280, 3) | 52.402 | 53.368 | 445.31 | 72.73 | [Link](https://github.com/ultralytics/yolov5/)| |
| yolov5xu | (640, 640, 3) | 52.600 | 53.090 | 246.55 | 97.23 | [Link](https://docs.ultralytics.com/models/yolov5/)| |
| yolov5x6 | (1280, 1280, 3) | 53.706 | 54.706 | 838.81 | 140.73 | [Link](https://github.com/ultralytics/yolov5/)| |
| yolov7 | (640,640,3) | 50.402 | 50.941 | 104.67 | 36.91 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| yolov7x | (640,640,3) | 52.103 | 52.706 | 189.88 | 71.31 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| yolov8s | (640,640,3) | 44.062 | 44.918 | 28.64 | 11.16 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| yolov8m | (640,640,3) | 49.657 | 50.240 | 79.00 | 25.89 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| yolov8l | (640,640,3) | 52.340 | 52.773 | 165.24 | 43.67 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| yolov8x | (640,640,3) | 53.365 | 53.802 | 257.92 | 68.20 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| yolov9m | (640,640,3) | 50.351 | 51.191 | 76.43 | 19.98 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| yolov9c | (640,640,3) | 52.349 | 52.917 | 102.34 | 25.29 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| yolo11s | (640,640,3) | 45.733 | 46.617 | 21.69 | 9.44 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| yolo11m | (640,640,3) | 50.638 | 51.310 | 68.24 | 20.09 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| yolo11l | (640,640,3) | 52.468 | 53.165 | 87.37 | 25.34 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| yolo11x | (640,640,3) | 54.059 | 54.478 | 195.67 | 56.92 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| yolo12s | (640,640,3) | 46.746 | 47.687 | 23.84 | 9.26 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| yolo12m | (640,640,3) | 51.568 | 52.297 | 71.91 | 20.17 | [Link](https://docs.ultralytics.com/models/yolo12/) | |

## Instance Segmentation (COCO)

| Model | Input Size <br> (H, W, C)| mAPmask <br> (NPU) | mAPmask <br> (GPU)| FLOPs (B) | params (M) |Source| Note |
|-------|--------------------------|--------------------|-------------------|-----------|------------|------|------|
| yolov5l-seg | (640,640,3) | 39.041 | 39.784 | 147.83 | 47.89 | [Link](https://github.com/ultralytics/yolov5/) | |
| yolov5x-seg | (640,640,3) | 40.697 | 41.137 | 265.81 | 88.77 | [Link](https://github.com/ultralytics/yolov5/) | |
| yolov8s-seg | (640,640,3) | 36.072 | 36.582 | 42.64 | 11.81 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| yolov8m-seg | (640,640,3) | 39.978 | 40.444 | 110.26 | 27.27 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| yolov8l-seg | (640,640,3) | 41.993 | 42.316 | 220.55 | 45.97 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| yolov8x-seg | (640,640,3) | 42.913 | 43.020 | 344.20 | 71.78 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| yolov9c-seg | (640,640,3) | 42.249 | 42.460 | 145.72 | 27.45 | [Link](https://docs.ultralytics.com/models/yolov9/) | |
| yolo11s-seg | (640,640,3) | 36.874 | 37.602 | 35.69 | 10.10 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| yolo11m-seg | (640,640,3) | 41.031 | 41.546 | 123.56 | 22.40 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| yolo11l-seg | (640,640,3) | 42.509 | 42.831 | 142.68 | 27.65 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| yolo11x-seg | (640,640,3) | 43.614 | 43.746 | 319.78 | 62.09 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
