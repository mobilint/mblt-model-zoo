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
# Currently, default is "DEFAULT", or "IMAGENET1K_V1
resnet50 = ResNet50(model_type = "IMAGENET1K_V2")

# Download the model to local directory and load it
resnet50 = ResNet50(local_path = "path/to/local/")

# Load the model from a local path or download as filename and file path you want
resnet50 = ResNet50(local_path = "path/to/local/model.mxq")

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

| Model | Input Size <br> (H, W, C)|Top1 Acc <br> (NPU)| Top1 Acc <br> (GPU)| Ops (G) | MACs |Source|
|------------|------------|-----------|--------------------|--------|-------|------|
alexnet	        | (224,224,3)	| 56.01	| 56.56	| 1.42	| 0.71	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html)	
densenet121	        | (224,224,3)	| 73.86	| 74.44	| 5.70	| 2.85	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html)	
densenet161	        | (224,224,3)	| 76.69	| 77.11	| 15.52	| 7.76	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet161.html)	
densenet169	        | (224,224,3)	| 74.90	| 75.61	| 6.76	| 3.38	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet169.html)	
densenet201	        | (224,224,3)	| 76.30	| 76.89	| 8.64	| 4.32	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet201.html)	
efficientnet_b1	    | (240,240,3)	| 77.22	| 78.60	| 1.39	| 0.69	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b1.html)	
mnasnet0_5	        | (224,224,3)	| 67.01	| 67.73	| 0.20	| 0.10	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.mnasnet0_5.html)	
mnasnet0_75	        | (224,224,3)	| 70.42	| 71.18	| 0.43	| 0.21	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.mnasnet0_75.html)	
mnasnet1_0  	    | (224,224,3)	| 73.06	| 73.47	| 0.62	| 0.31	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.mnasnet1_0.html)	
mobilenet_v1	                | (224,224,3)	| 72.35	| 70.60	| 1.14	| 0.57	| [Link](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)	
mobilenet_v2	    | (224,224,3)	| 72.85	| 71.87	| 0.60	| 0.30	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html)	
regnet_x_16gf	    | (224,224,3)	| 79.83	| 80.06	| 31.88	| 15.94	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_16gf.html)	
regnet_x_1_6gf	    | (224,224,3)	| 76.84	| 77.05	| 3.20	| 1.60	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_1_6gf.html)	
regnet_x_32gf	    | (224,224,3)	| 80.46	| 80.61	| 63.47	| 31.73	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_32gf.html)	
regnet_x_3_2gf	    | (224,224,3)	| 78.10	| 78.36	| 6.35	| 3.17	| [Link](https://pytorch.org/vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html)	
regnet_x_400mf	    | (224,224,3)	| 72.37	| 72.83	| 0.82	| 0.41	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_400mf.html)	
regnet_x_800mf	    | (224,224,3)	| 74.94	| 75.22	| 1.60	| 0.80	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_800mf.html)	
regnet_x_8gf	    | (224,224,3)	| 79.21	| 79.34	| 15.99	| 7.99	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_8gf.html)	
resnet18	        | (224,224,3)	| 69.54	| 69.75	| 3.63	| 1.81	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)	
resnet34	        | (224,224,3)	| 73.08	| 73.30	| 7.33	| 3.66	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html)	
resnet50_v1 	    | (224,224,3)	| 75.92	| 76.13	| 8.18	| 4.09	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)	
resnet50_v2	        | (224,224,3)	| 80.25	| 80.86	| 8.18	| 4.09	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)	
resnet101	        | (224,224,3)	| 77.06	| 77.37	| 15.60	| 7.80	| [Link](https://pytorch.org/vision/2.0/models/generated/torchvision.models.resnet101.html)	
resnet152	        | (224,224,3)	| 77.82	| 78.31	| 23.04	| 11.52	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html)	
resnext50_32x4d	    | (224,224,3)	| 77.48	| 77.61	| 8.46	| 4.23	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnext50_32x4d.html)	
resnext101_32x8d	| (224,224,3)	| 79.01	| 79.31	| 32.83	| 16.41	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnext101_32x8d.html)	
resnext101_64x4d	| (224,224,3)	| 82.77	| 83.25	| 30.92	| 15.46	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnext101_64x4d.html)	
shufflenet_v2_x1_0	| (224,224,3)	| 68.74	| 69.36	| 0.62	| 0.31	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.shufflenet_v2_x1_0.html)	
shufflenet_v2_x1_5	| (224,224,3)	| 72.41	| 72.98	| 1.36	| 0.68	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.shufflenet_v2_x1_5.html)	
shufflenet_v2_x2_0	| (224,224,3)	| 75.38	| 76.23	| 2.65	| 1.32	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.shufflenet_v2_x2_0.html)	
vgg11   	        | (224,224,3)	| 68.82	| 69.04| 15.22	| 7.61	| [Link](https://pytorch.org/vision/master/models/generated/torchvision.models.vgg11.html)	
vgg11_bn   	        | (224,224,3)	| 70.02	| 70.37	| 15.22	| 7.61	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg11_bn.html)	
vgg13	            | (224,224,3)	| 69.65	| 69.93	| 22.62	| 11.31	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg13.html)	
vgg13_bn	        | (224,224,3)	| 71.25	| 71.59	| 22.62	| 11.31	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg13_bn.html)	
vgg16   	        | (224,224,3)	| 71.41	| 71.59	| 30.94	| 15.47	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg16.html)	
vgg16_bn   	        | (224,224,3)	| 73.18	| 73.36	| 30.94	| 15.47	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg16_bn.html)	
vgg19	            | (224,224,3)	| 72.27	| 72.38	| 39.26	| 19.63	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg19.html)	
vgg19_bn	        | (224,224,3)	| 73.90	| 74.22	| 39.26	| 19.63	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg19_bn.html)	

### Object Detection (COCO)
| Model | Input Size <br> (H, W, C)| mAP <br> (NPU) | mAP <br> (GPU)| Ops (G) | MACs |Source|
|------------|------------|-----------|--------------------|--------|-------|------|
yolov7	                | (640,640,3)	| 50.13	| 51.14	| 104.66	| 52.33	| [Link](https://github.com/WongKinYiu/yolov7)	
yolov8s	                        | (640,640,3)	| 44.07	| 44.95	| 28.64	| 14.32	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)	
yolov8m	                        | (640,640,3)	| 49.68	| 50.22	| 79.00	| 39.50	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)
yolov8l	                        | (640,640,3)	| 52.31	| 52.75	| 165.24	| 82.62	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)	
yolov8x	                        | (640,640,3)	| 53.37	| 53.90	| 257.92	| 128.96	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)
yolov9m	                        | (640,640,3)	| 50.65	| 51.40	| 76.95	| 38.47	| [Link](https://github.com/WongKinYiu/yolov9)		
yolov9c	                        | (640,640,3)	| 52.16	| 52.68	| 102.86	| 51.43	| [Link](https://github.com/WongKinYiu/yolov9)	

### Instance Segmentation (COCO)
| Model | Input Size <br> (H, W, C)| mAPmask <br> (NPU) | mAPmask <br> (GPU)| Ops (G) | MACs |Source|
|------------|------------|-----------|--------------------|--------|-------|------|
yolov5l-seg	                    | (640,640,3)	| 39.32	| 39.67	| 147.83	| 73.91	| [Link](https://github.com/ultralytics/yolov5/releases)	
yolov8s-seg	                    | (640,640,3)	| 35.90	| 36.50	| 42.64	| 21.32	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#key-features-of-yolov8)	
yolov8m-seg                 	| (640,640,3)	| 39.88	| 40.40	|  110.26	| 55.13	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)	
yolov8l-seg	                    | (640,640,3)	| 42.04	| 42.27	| 220.55	| 110.27	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)


## License
The Mobilint Model Zoo is released under BSD 3-Clause License. Please see the [LICENSE](https://github.com/mobilint/mblt-model-zoo/blob/master/LICENSE) file for more details.

## Support & Issues
If you encounter any problem with this package, please feel free to contact [us](mailto:tech-support@mobilint.com).