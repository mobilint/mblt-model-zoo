Mobilint Model Zoo
========================

<div align="center">
<p>
 <a href="https://www.mobilint.com/" target="_blank">
<img src="https://raw.githubusercontent.com/mobilint/mblt-model-zoo/master/assets/Mobilint_Logo_Primary.png" alt="Mobilint Logo" width="60%">
</a>
</p>
</div>

**mblt-model-zoo** is a curated collection of AI models optimized by [Mobilint](https://www.mobilint.com/)’s Neural Processing Units (NPUs).

Designed to help developers accelerate deployment, Mobilint's Model Zoo offers access to public, pre-trained, and pre-quantized models for vision, language, and multimodal tasks. Along with performance results, we provide pre- and post-processing tools to help developers evaluate, fine-tune, and integrate the models with ease.

## Installation
- Install Mobilint ACCELerator(MACCEL) on your environment. In case you are not Mobilint customer, please contact [us](mailto:tech-support@mobilint.com).
- Install **mblt-model-zoo** using pip:
```bash
pip install mblt-model-zoo
```
- If you want to install the latest version from the source, clone the repository and install it:
```bash
git clone https://github.com/mobilint/mblt-model-zoo.git
cd mblt-model-zoo
pip install -e .
```
## Quick Start Guide
### Initializing Quantized Model Class
**mblt-model-zoo** provides a quantized model with associated pre- and post-processing tools. The following code snippet shows how to use the pre-trained model for inference.

```python
from mblt_model_zoo.vision import ResNet50

# Load the pre-trained model. 
# Automatically download the model if not found in the local cache.
resnet50 = ResNet50() 

# Load the model trained with different recipe
# Currently, default is "DEFAULT", or "IMAGENET1K_V1.
resnet50 = ResNet50(model_type = "IMAGENET1K_V2")

# Download the model to local directory and load it
resnet50 = ResNet50(local_path = "path/to/local/") # the file will be downloaded to "path/to/local/model.mxq"

# Load the model from a local path or download as filename and file path you want
resnet50 = ResNet50(local_path = "path/to/local/model.mxq")

# Set inference mode for better performance
# Aries supports "single", "multi" and "global" inferece mode. Default is "global"
resnet50 = ResNet50(infer_mode = "global")

# (Beta) If you are holding a model compiled for Regulus, enable inference on the Regulus device.
resnet50 = ResNet50(product = "regulus")

# In summary, the model can be loaded with the following arguments. 
# You may customize those arguments to work with Mobilint's NPU.
resnet50 = ResNet50(
    local_path = None,
    model_type = "DEFAULT",
    infer_mode = "global",
    product = "aries",
)

```
### Working with Quantized Model
With the image given as path, PIL image, numpy array, or torch tensor, you can perform inference with the quantized model. The following code snippet shows how to use the quantized model for inference:
```python
image_path = "path/to/image.jpg"

input_img = resnet50.preprocess(image_path) # Preprocess the input image
output = resnet50(input_img) # Perform inference with the quantized model
result = resnet50.postprocess(output) # Postprocess the output

result.plot(
    source_path=image_path,
    save_path="path/to/save/result.jpg",
)
```
### Listing Available Models
**mblt-model-zoo** offers a function to list all available models. You can use the following code snippet to list the models for a specific task (e.g., image classification, object detection, etc.):

```python
from mblt_model_zoo.vision import list_models
from pprint import pprint

available_models = list_models()
pprint(available_models)
```

## Model List
The following tables summarize the models available in **mblt-model-zoo**. We provide the models that are quantized with our advanced quantization techniques.

### Image Classification (ImageNet)

| Model | Input Size <br> (H, W, C)|Top1 Acc <br> (NPU)| Top1 Acc <br> (GPU)| FLOPs (B) | params (M) |Source|Note|
|------------|------------|-----------|--------------------|--------|-------|------|------|
AlexNet | (224,224,3)| 56.076 | 56.552 | 0.71 | 61.10 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html) |  |
ConvNeXt_Tiny | (224,224,3) | 82.242 | 82.460| 4.46 | 28.59 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html) |  |
ConvNeXt_Small | (224,224,3) | 83.182 | 83.560 | 8.68 | 50.22 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_small.html) |  |
ConvNeXt_Base | (224,224,3) | 83.834 | 84.050 | 15.36 | 88.59 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html) |  |
DenseNet121 | (224,224,3) | 73.948 | 74.414 | 2.83 | 7.98 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html) |  |
DenseNet161 | (224,224,3) | 76.890 | 77.132 | 7.73 | 28.68 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet161.html) |  |
DenseNet169 | (224,224,3) | 74.984 | 75.566 | 3.36 | 14.15 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet169.html) |  |
DenseNet201 | (224,224,3) | 76.194 | 76.880 | 4.29 | 20.01 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet201.html) |  |
GoogLeNet | (224,224,3) | 69.566 | 69.780 | 1.50 | 6.62 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.googlenet.html)	|  |
Inception_V3 | (299,299,3) | 77.110 | 77.278 | 5.71 | 27.16 | [Link](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html)|  |
MNASNet1_0 | (224,224,3)| 72.752 | 73.422 | 0.31 | 4.38 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mnasnet1_0.html) | |
MNASNet1_3 | (224,224,3) | 75.708 | 76.466 | 0.53 | 6.28 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mnasnet1_3.html) | |
RegNet_X_400MF | (224,224,3) | 72.574 | 72.900 | 0.41 | 5.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_400mf.html) | IMAGENET1K_V1 |
RegNet_X_400MF | (224,224,3) | 74.212 | 74.846 | 0.41 | 5.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_400mf.html) | IMAGENET1K_V2 |
RegNet_X_800MF | (224,224,3) | 74.920 | 75.204 | 0.80 | 7.26 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_800mf.html) | IMAGENET1K_V1 |
RegNet_X_800MF | (224,224,3) | 76.868 | 77.488 | 0.80 | 7.26 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_800mf.html) | IMAGENET1K_V2 |
RegNet_X_1_6GF | (224,224,3) | 76.818 | 77.080 | 1.60 | 9.19 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_1_6gf.html) | IMAGENET1K_V1 |
RegNet_X_1_6GF | (224,224,3) | 79.220 | 79.670 | 1.60 | 9.19 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_1_6gf.html) | IMAGENET1K_V2 |
RegNet_X_3_2GF | (224,224,3) | 78.040 | 78.346 | 3.18 | 15.30 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html) | IMAGENET1K_V1 |
RegNet_X_3_2GF | (224,224,3) | 80.712 | 81.188 | 3.18 | 15.30 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html) | IMAGENET1K_V2 |
RegNet_X_8GF | (224,224,3) | 79.248 | 79.368 | 8.00 | 39.57 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_8gf.html) | IMAGENET1K_V1 |
RegNet_X_8GF | (224,224,3) | 81.330 | 81.680 | 8.00 | 39.57 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_8gf.html) | IMAGENET1K_V2 |
RegNet_X_16GF | (224,224,3) | 79.880 | 80.090 | 15.94 | 54.28 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_16gf.html) | IMAGENET1K_V1 |
RegNet_X_16GF | (224,224,3) | 82.334 | 82.712 | 15.94 | 54.28 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_16gf.html) | IMAGENET1K_V2 |
RegNet_X_32GF | (224,224,3) | 80.420 | 80.592 | 31.74 | 107.81 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_32gf.html) | IMAGENET1K_V1 |
RegNet_X_32GF | (224,224,3) | 82.798 | 83.014 | 31.74 | 107.81 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_32gf.html) | IMAGENET1K_V2 |
RegNet_Y_400MF | (224,224,3) | 73.528 | 73.998 | 0.40 | 4.34 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html) | IMAGENET1K_V1 |
RegNet_Y_400MF | (224,224,3) | 75.156 | 75.804 | 0.40 | 4.34 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html) | IMAGENET1K_V2 |
RegNet_Y_800MF | (224,224,3) | 76.132 | 76.406 | 0.83 | 6.43 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_800mf.html) | IMAGENET1K_V1|
RegNet_Y_800MF | (224,224,3) | 78.266 | 78.904 | 0.83 | 6.43 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_800mf.html) | IMAGENET1K_V2|
RegNet_Y_1_6GF | (224,224,3) | 77.342 | 77.934 | 1.61 | 11.20 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_1_6gf.html) | IMAGENET1K_V1 |
RegNet_Y_1_6GF | (224,224,3) | 80.404 | 80.876 | 1.61 | 11.20 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_1_6gf.html) | IMAGENET1K_V2 |
RegNet_Y_3_2GF | (224,224,3) | 78.474 | 78.962 | 3.18 | 19.44 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_y_3_2gf.html) | IMAGENET1K_V1 |
RegNet_Y_3_2GF | (224,224,3) | 81.374 | 82.010 | 3.18 | 19.44 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_y_3_2gf.html) | IMAGENET1K_V2 |
RegNet_Y_8GF | (224,224,3) | 79.792 | 80.052 | 8.47 | 39.38 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_8gf.html) | IMAGENET1K_V1 |
RegNet_Y_8GF | (224,224,3) | 82.438 | 82.822 | 8.47 | 39.38 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_8gf.html) | IMAGENET1K_V2 |
RegNet_Y_16GF | (224,224,3) | 80.110 | 80.428 | 15.91 | 83.59 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_16gf.html)| IMAGENET1K_V1 |
RegNet_Y_16GF | (224,224,3) | 82.314 | 82.868 | 15.91 | 83.59 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_16gf.html)| IMAGENET1K_V2 |
RegNet_Y_32GF | (224,224,3) | 80.532 | 80.840 | 32.28 | 145.05 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_32gf.html) | IMAGENET1K_V1 |
RegNet_Y_32GF | (224,224,3) | 82.798 | 83.362 | 32.28 | 145.05 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_32gf.html) | IMAGENET1K_V2 |
ResNet18 | (224,224,3) | 69.552 | 69.774 | 1.81 | 11.69 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet18.html) | |
ResNet34 | (224,224,3) | 73.220 | 73.310 | 3.66 | 21.80 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet34.html) | |
ResNet50 | (224,224,3) | 75.940 | 76.128 | 4.09 | 25.56 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet50.html) | IMAGENET1K_V1 |
ResNet50 | (224,224,3) | 80.390 | 80.844 | 4.09 | 25.56 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet50.html) | IMAGENET1K_V2 |
ResNet101 | (224,224,3) | 77.140 | 77.362 | 7.80 | 44.55 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet101.html) | IMAGENET1K_V1 |
ResNet101 | (224,224,3) | 81.538 | 81.918 | 7.80 | 44.55 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet101.html) | IMAGENET1K_V2 |
ResNet152 | (224,224,3) | 77.860 | 78.316 | 11.51 | 60.19 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet152.html) | IMAGENET1K_V1 |
ResNet152 | (224,224,3) | 81.888 | 82.272 | 11.51 | 60.19 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet152.html) | IMAGENET1K_V2 |
ResNeXt50_32X4D | (224,224,3) | 77.560 | 77.630 | 4.23 | 25.03 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext50_32x4d.html) | IMAGENET1K_V1 |
ResNeXt50_32X4D | (224,224,3) | 80.744 | 81.220 | 4.23 | 25.03 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext50_32x4d.html) | IMAGENET1K_V2 |
ResNeXt101_32X8D | (224,224,3) | 78.974 | 79.288 | 16.41 | 88.79 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_32x8d.html) | IMAGENET1K_V1 |
ResNeXt101_32X8D | (224,224,3) | 82.516 | 82.780 | 16.41 | 88.79 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_32x8d.html) | IMAGENET1K_V2 |
ResNeXt101_64X4D | (224,224,3) | 82.900 | 83.236 | 15.46 | 83.46 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_64x4d.html) | |
ShuffleNet_V2_X1_0 | (224,224,3) | 68.788 | 69.312 | 0.14 | 2.28 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_0.html) | |
ShuffleNet_V2_X1_5 | (224,224,3) | 72.240 | 72.966 | 0.30 | 3.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_5.html) | |
ShuffleNet_V2_X2_0 | (224,224,3) | 75.638 | 76.222 | 0.58 | 7.39 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x2_0.html) | |
VGG11 | (224,224,3) | 68.614 | 68.978 | 7.61 | 132.86 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg11.html) | |
VGG11_BN | (224,224,3) | 69.978 | 70.328 | 7.61 | 132.87 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg11_bn.html) | |
VGG13 | (224,224,3) | 69.588 | 69.886 | 11.31 | 133.05 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg13.html) | |
VGG13_BN | (224,224,3) | 71.268 | 71.570 | 11.31 | 133.05 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg13_bn.html) | |
VGG16 | (224,224,3) | 71.344 | 71.610 | 15.47 | 138.36 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg16.html) | |
VGG16_BN | (224,224,3) | 73.184 | 73.408 | 15.47 | 138.37 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg16_bn.html) | |
VGG19 | (224,224,3) | 72.142 | 72.384 | 19.63 | 143.67 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg19.html) | |
VGG19_BN | (224,224,3) | 73.974 | 74.166 | 19.63 | 143.68 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg19_bn.html) | |
Wide_ResNet50_2 | (224,224,3) | 78.364 | 78.490 | 11.40 | 68.88 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html) | IMAGENET1K_V1 |
Wide_ResNet50_2 | (224,224,3) | 81.230 | 81.626 | 11.40 | 68.88 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html) | IMAGENET1K_V2 |
Wide_ResNet101_2 | (224,224,3) | 78.478 | 78.834 | 22.75 | 126.89 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet101_2.html) | IMAGENET1K_V1 |
Wide_ResNet101_2 | (224,224,3) | 82.230 | 82.504 | 22.75 | 126.89 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet101_2.html) | IMAGENET1K_V2 |

### Object Detection (COCO)

| Model | Input Size <br> (H, W, C)| mAP <br> (NPU) | mAP <br> (GPU)| FLOPs (B) | params (M) | Source | Note |
|------------|------------|-----------|--------------------|--------|-------|------|----|
yolov3u | (640,640,3) | 51.085 | 51.577 | 282.68  | 103.73 | [Link](https://docs.ultralytics.com/models/yolov3/)| w/o Anchor |
yolov3-sppu | (640,640,3) | 51.311 | 51.763 | 283.57 | 104.78 | [Link](https://docs.ultralytics.com/models/yolov3/)| w/o Anchor |
yolov5su | (640, 640, 3) | 42.050 | 42.871 | 24.03 | 9.14 | [Link](https://docs.ultralytics.com/models/yolov5/)| w/o Anchor |
yolov5mu | (640, 640, 3) | 48.211 | 48.906 | 64.27 | 25.09 | [Link](https://docs.ultralytics.com/models/yolov5/)| w/o Anchor |
yolov5lu | (640, 640, 3) | 51.721 | 52.171 | 135.09 | 53.19 | [Link](https://docs.ultralytics.com/models/yolov5/)| w/o Anchor |
yolov5xu | (640, 640, 3) | 52.592 | 53.090 | 246.55 | 97.23 | [Link](https://docs.ultralytics.com/models/yolov5/)| w/o Anchor |
yolov7 | (640,640,3) | 50.032 | 50.941 | 104.67 | 36.91 | [Link](https://github.com/WongKinYiu/yolov7/)| w/ Anchor |
yolov7x | (640,640,3) | 51.697 | 52.706 | 189.88 | 71.31 | [Link](https://github.com/WongKinYiu/yolov7/)| w/ Anchor |
yolov8s | (640,640,3) | 44.022 | 44.918 | 28.64 | 11.16 | [Link](https://docs.ultralytics.com/models/yolov8/) | w/o Anchor |
yolov8m | (640,640,3) | 49.687 | 50.240 | 79.00 | 25.89 | [Link](https://docs.ultralytics.com/models/yolov8/) | w/o Anchor |
yolov8l | (640,640,3) | 52.334 | 52.773 | 165.24 | 43.67 | [Link](https://docs.ultralytics.com/models/yolov8/) | w/o Anchor |
yolov8x | (640,640,3) | 53.374 | 53.802 | 257.92 | 68.20 | [Link](https://docs.ultralytics.com/models/yolov8/) | w/o Anchor |
yolov9m | (640,640,3) | 50.390 | 51.191 | 76.43 | 19.98 | [Link](https://github.com/WongKinYiu/yolov9/) | w/o Anchor |
yolov9c | (640,640,3) | 52.414 | 52.917 | 102.34 | 25.29 | [Link](https://github.com/WongKinYiu/yolov9/) | w/o Anchor |

### Instance Segmentation (COCO)

| Model | Input Size <br> (H, W, C)| mAPmask <br> (NPU) | mAPmask <br> (GPU)| FLOPs (B) | params (M) |Source| Note|
|------------|------------|-----------|--------------------|--------|-------|------|-----|
yolov5m-seg | (640,640,3) | 36.314 | 36.977 | 70.91 | 22.97 | [Link](https://github.com/ultralytics/yolov5/) | w/ Anchor |
yolov5l-seg | (640,640,3) | 39.087 | 39.784 | 147.83 | 47.89 | [Link](https://github.com/ultralytics/yolov5/) | w/ Anchor |
yolov5x-seg | (640,640,3) | 40.753 | 41.137 | 265.81 | 88.77 | [Link](https://github.com/ultralytics/yolov5/) | w/ Anchor |
yolov8s-seg | (640,640,3) | 36.139 | 36.582 | 42.64 | 11.81 | [Link](https://docs.ultralytics.com/models/yolov8/#key-features-of-yolov8) | w/o Anchor |
yolov8m-seg | (640,640,3) | 40.032 | 40.444 | 110.26 | 27.27 | [Link](https://docs.ultralytics.com/models/yolov8/#overview) | w/o Anchor |
yolov8l-seg | (640,640,3) | 41.877 | 42.316 | 220.55 | 45.97 | [Link](https://docs.ultralytics.com/models/yolov8/) | w/o Anchor |
yolov8x-seg | (640,640,3) | 42.717 | 43.020 | 344.20 | 71.78 | [Link](https://docs.ultralytics.com/models/yolov8/) | w/o Anchor |
yolov9c-seg | (640,640,3) | 42.236 | 42.460 | 145.72 | 27.45 | [Link](https://docs.ultralytics.com/models/yolov9/) | w/o Anchor |

## Optional Extras
When working with tasks other than vision, extra dependencies may be required. Those options can be installed via `pip install mblt-model-zoo[NAME]` or `pip install -e .[NAME]`.

Currently, this optional functions are only available on environment equipped with Mobilint's [Aries](https://www.mobilint.com/aries).

|Name|Use|
|-------|------|
|transformers|For using HuggingFace transformers related models| 

## License
The Mobilint Model Zoo is released under BSD 3-Clause License. Please see the [LICENSE](https://github.com/mobilint/mblt-model-zoo/blob/master/LICENSE) file for more details.

## Support & Issues
If you encounter any problem with this package, please feel free to contact [us](mailto:tech-support@mobilint.com).