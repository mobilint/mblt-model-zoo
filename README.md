Mobilint Model Zoo
========================

<div align="center">
<p>
 <a href="https://www.mobilint.com/" target="_blank">
<img src="assets/Mobilint_Logo_Primary.png" width="60%"/>
</a>
</p>
</div>

`mblt-model-zoo` is an open model zoo for [Mobilint](https://www.mobilint.com/) NPU. It provides a collection of public pre-trained, pre-quantized models, and pre/post-processing tools associated with the quantized models.

## Model List
The table summarizes all models available in `mblt-model-zoo`. 

| Model | Input Size | NPU Metric (Acc/mAP) | GPU Metric|Relative Performance(%)| Ops(G) | MACs |Source|
|------------|------------|----------------------|-----------|--------------------|--------|-------|------|
alexnet_torchvision 	        | (224,224,3)	| 56.01	| 56.56	| 99.02	| 1.42	| 0.71	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html)	
vgg11_torchvision   	        | (224,224,3)	| 68.82	| 69.04	| 99.68	| 15.22	| 7.61	| [Link](https://pytorch.org/vision/master/models/generated/torchvision.models.vgg11.html)	
vgg11_bn_torchvision   	        | (224,224,3)	| 70.02	| 70.37	| 99.50	| 15.22	| 7.61	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg11_bn.html)	
vgg13_torchvision	            | (224,224,3)	| 69.65	| 69.928	| 99.60	| 22.62	| 11.31	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg13.html)	
vgg13_bn_torchvision	        | (224,224,3)	| 71.25	| 71.586	| 99.53	| 22.62	| 11.31	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg13_bn.html)	
vgg16_torchvision   	        | (224,224,3)	| 71.41	| 71.592	| 99.74	| 30.94	| 15.47	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg16.html)	
vgg16_bn_torchvision   	        | (224,224,3)	| 73.18	| 73.36	| 99.75	| 30.94	| 15.47	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg16_bn.html)	
vgg19_torchvision	            | (224,224,3)	| 72.27	| 72.376	| 99.85	| 39.26	| 19.63	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg19.html)	
vgg19_bn_torchvision	        | (224,224,3)	| 73.9	| 74.218	| 99.57	| 39.26	| 19.63	| [Link](https://pytorch.org/vision/0.20/models/generated/torchvision.models.vgg19_bn.html)	
densenet121_torchvision	        | (224,224,3)	| 73.86	| 74.44	| 99.22	| 5.70	| 2.85	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html)	
densenet161_torchvision	        | (224,224,3)	| 76.69	| 77.11	| 99.45	| 15.52	| 7.76	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet161.html)	
densenet169_torchvision	        | (224,224,3)	| 74.9	| 75.61	| 99.06	| 6.76	| 3.38	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet169.html)	
densenet201_torchvision	        | (224,224,3)	| 76.3	| 76.89	| 99.23	| 8.64	| 4.32	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet201.html)	
efficientnet_b1_torchvision	    | (240,240,3)	| 77.22	| 78.6	| 98.24	| 1.39	| 0.69	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b1.html)	
mnasnet0_5_torchvision	        | (224,224,3)	| 67.01	| 67.73	| 98.93	| 0.20	| 0.10	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.mnasnet0_5.html)	
mnasnet0_75_torchvision	        | (224,224,3)	| 70.42	| 71.18	| 98.93	| 0.43	| 0.21	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.mnasnet0_75.html)	
mnasnet1_0_torchvision  	    | (224,224,3)	| 73.06	| 73.47	| 99.44	| 0.62	| 0.31	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.mnasnet1_0.html)	
mobilenet_v1	                | (224,224,3)	| 72.35	| 70.6	| 102.47	| 1.14	| 0.57	| [Link](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)	
mobilenet_v2_torchvision	    | (224,224,3)	| 72.85	| 71.87	| 101.36	| 0.60	| 0.30	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html)	
regnet_x_16gf_torchvision	    | (224,224,3)	| 79.83	| 80.06	| 99.71	| 31.88	| 15.94	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_16gf.html)	
regnet_x_1_6gf_torchvision	    | (224,224,3)	| 76.84	| 77.05	| 99.72	| 3.20	| 1.60	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_1_6gf.html)	
regnet_x_32gf_torchvision	    | (224,224,3)	| 80.46	| 80.61	| 99.81	| 63.47	| 31.73	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_32gf.html)	
regnet_x_3_2gf_torchvision	    | (224,224,3)	| 78.1	| 78.36	| 99.66	| 6.35	| 3.17	| [Link](https://pytorch.org/vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html)	
regnet_x_400mf_torchvision	    | (224,224,3)	| 72.37	| 72.83	| 99.36	| 0.82	| 0.41	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_400mf.html)	
regnet_x_800mf_torchvision	    | (224,224,3)	| 74.94	| 75.22	| 99.62	| 1.60	| 0.80	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_800mf.html)	
regnet_x_8gf_torchvision	    | (224,224,3)	| 79.21	| 79.34	| 99.83	| 15.99	| 7.99	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_x_8gf.html)	
resnet18_torchvision	        | (224,224,3)	| 69.54	| 69.75	| 99.69	| 3.63	| 1.81	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)	
resnet34_torchvision	        | (224,224,3)	| 73.08	| 73.3	| 99.69	| 7.33	| 3.66	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html)	
resnet50_v1_torchvision 	    | (224,224,3)	| 75.92	| 76.13	| 99.72	| 8.18	| 4.09	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)	
resnet50_v2_torchvision	        | (224,224,3)	| 80.25	| 80.86	| 99.24	| 8.18	| 4.09	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)	
resnet101_torchvision	        | (224,224,3)	| 77.06	| 77.374	| 99.59	| 15.6	| 7.80	| [Link](https://pytorch.org/vision/2.0/models/generated/torchvision.models.resnet101.html)	
resnet152_torchvision	        | (224,224,3)	| 77.82	| 78.312	| 99.37	| 23.04	| 11.52	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html)	
resnext50_32x4d_torchvision	    | (224,224,3)	| 77.48	| 77.61	| 99.83	| 8.46	| 4.23	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnext50_32x4d.html)	
resnext101_32x8d_torchvision	| (224,224,3)	| 79.01	| 79.31	| 99.62	| 32.83	| 16.41	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnext101_32x8d.html)	
resnext101_64x4d_torchvision	| (224,224,3)	| 82.77	| 83.25	| 99.42	| 30.92	| 15.46	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnext101_64x4d.html)	
shufflenet_v2_x1_0_torchvision	| (224,224,3)	| 68.74	| 69.36	| 99.10	| 0.62	| 0.31	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.shufflenet_v2_x1_0.html)	
shufflenet_v2_x1_5_torchvision	| (224,224,3)	| 72.41	| 72.98	| 99.21	| 1.36	| 0.68	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.shufflenet_v2_x1_5.html)	
shufflenet_v2_x2_0_torchvision	| (224,224,3)	| 75.38	| 76.23	| 98.88	| 2.65	| 1.32	| [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.shufflenet_v2_x2_0.html)	
yolov5l-seg	                    | (640,640,3)	| 39.318	| 39.67	| 99.11	| 147.83	| 73.91	| [Link](https://github.com/ultralytics/yolov5/releases)	
yolov7_640_640	                | (640,640,3)	| 50.134	| 51.14	| 98.03	| 104.66	| 52.33	| [Link](https://github.com/WongKinYiu/yolov7)	
yolov8l-seg	                    | (640,640,3)	| 42.042	| 42.27	| 99.46	| 220.55	| 110.27	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)
yolov8l	                        | (640,640,3)	| 52.31	| 52.75	| 99.16	| 165.24	| 82.62	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)	
yolov8m-seg                 	| (640,640,3)	| 39.882	| 40.4	| 98.71	| 110.26	| 55.13	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)	
yolov8m	                        | (640,640,3)	| 49.676	| 50.22	| 98.91	| 79.00	| 39.50	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)	
yolov8s-seg	                    | (640,640,3)	| 35.9	| 36.5	| 98.35	| 42.64	| 21.32	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#key-features-of-yolov8)	
yolov8s	                        | (640,640,3)	| 44.066	| 44.95	| 98.03	| 28.64	| 14.32	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)	
yolov8x	                        | (640,640,3)	| 53.371	| 53.9	| 99.01	| 257.92	| 128.96	| [Link](https://docs.ultralytics.com/ko/models/yolov8/#overview)	
yolov9c	                        | (640,640,3)	| 52.161	| 52.68	| 99.01	| 102.86	| 51.43	| [Link](https://github.com/WongKinYiu/yolov9)	
yolov9m	                        | (640,640,3)	| 50.648	| 51.4	| 98.53	| 76.95	| 38.47	| [Link](https://github.com/WongKinYiu/yolov9)	

## Support & Issues
If you encounter any problem with this package, please feel free to contact [us](tech-support@mobilint.com).