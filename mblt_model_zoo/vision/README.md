# Pre-Trained Vision Models

Here, we give the full list of publicly pre-trained models supported by the Mobilint Model Zoo.

Further usage examples can be found in the [tests](../../tests/vision) directory.

## Image Classification

| Model | Input Size<br>(H,W,C) | Acc<sup>Top1</sup><br>(NPU) | Acc<sup>Top1</sup><br>(GPU) | FLOPs (B) | params (M) | Source | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AlexNet | (224,224,3) | 56.118 | 56.556 | 1.43 | 61.10 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html) | |
| CAFormer_S18 | (224,224,3) | 82.636 | 83.626 | 9.10 | 26.35 |  [Link](https://huggingface.co/timm/caformer_s18.sail_in1k) | sail_in1k |
| CAFormer_S36 | (224,224,3) | 82.764 | 84.440 | 17.67 | 39.32 |  [Link](https://huggingface.co/timm/caformer_s36.sail_in1k) | sail_in1k |
| CAFormer_M36 | (224,224,3) | 83.616 | 85.232 | 28.80 | 56.23 |  [Link](https://huggingface.co/timm/caformer_m36.sail_in1k) | sail_in1k |
| CAFormer_B36 | (224,224,3) | 84.342 | 85.482 | 49.41 | 98.78 |  [Link](https://huggingface.co/timm/caformer_b36.sail_in1k) | sail_in1k |
| CoAtNet_0_RW_224 | (224,224,3) | 81.080 | 82.418 | 10.06 | 30.46 |  [Link](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k) | sw_in1k |
| CoAtNet_1_RW_224 | (224,224,3) | 83.094 | 83.600 | 18.17 | 47.91 |  [Link](https://huggingface.co/timm/coatnet_1_rw_224.sw_in1k) | sw_in1k |
| CoAtNet_2_RW_224 | (224,224,3) | 84.952 | 86.534 | 32.80 | 82.11 |  [Link](https://huggingface.co/timm/coatnet_2_rw_224.sw_in12k_ft_in1k) | sw_in12k_ft_in1k |
| ConvFormer S18 | (224,224,3) | 81.946 | 82.866 | 8.59 | 26.77 | [Link](https://huggingface.co/timm/convformer_s18.sail_in1k) | sail_in1k |
| ConvFormer S36 | (224,224,3) | 83.216 | 84.016 | 16.66 | 40.01 |  [Link](https://huggingface.co/timm/caformer_s36.sail_in1k) | sail_in1k |
| ConvFormer M36 | (224,224,3) | 84.084 | 84.448 | 27.58 | 57.05 |  [Link](https://huggingface.co/timm/caformer_m36.sail_in1k) | sail_in1k |
| ConvFormer B36 | (224,224,3) | 84.384 | 84.830 | 49.79 | 99.88 |  [Link](https://huggingface.co/timm/caformer_b36.sail_in1k) | sail_in1k |
| ConvNeXt_Tiny | (224,224,3) | 82.352 | 82.458 | 9.11 | 28.59 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html) | |
| ConvNeXt_Small | (224,224,3) | 83.486 | 83.560 | 17.68 | 50.22 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_small.html) | |
| ConvNeXt_Base | (224,224,3) | 83.986 | 84.048 | 31.13 | 88.59 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html) | |
| ConvNeXt_Large | (224,224,3) | 84.348 | 84.410 | 69.35 | 197.77 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_large.html) | |
| DeiT_Tiny_Patch16_224 | (224,224,3) | 71.998 | 72.030 | 2.66 | 5.72 | [Link](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k) | fb_in1k |
| DeiT_Small_Patch16_224 | (224,224,3) | 79.806 | 79.790 | 9.50 | 22.05 | [Link](https://huggingface.co/timm/deit_small_patch16_224.fb_in1k) | fb_in1k |
| DeiT_Base_Patch16_224 | (224,224,3) | 81.932 | 81.980 | 35.73 | 86.57 | [Link](https://huggingface.co/timm/deit_base_patch16_224.fb_in1k) | fb_in1k |
| DeiT_Base_Patch16_384 | (384,384,3) | 83.108 | 83.100 | 115.00 | 86.86 | [Link](https://huggingface.co/timm/deit_base_patch16_384.fb_in1k) | fb_in1k |
| DeiT3_Small_Patch16_224 | (224,224,3) | 81.346 | 81.390 | 9.50 | 22.06 | [Link](https://huggingface.co/timm/deit3_small_patch16_224.fb_in1k) | fb_in1k |
| DeiT3_Small_Patch16_384 | (384,384,3) | 83.462 | 83.420 | 33.01 | 22.21 | [Link](https://huggingface.co/timm/deit3_small_patch16_384.fb_in1k) | fb_in1k |
| DeiT3_Medium_Patch16_224 | (224,224,3) | 83.034 | 83.044 | 16.39 | 38.85 | [Link](https://huggingface.co/timm/deit3_medium_patch16_224.fb_in1k) | fb_in1k |
| DeiT3_Base_Patch16_224 | (224,224,3) | 83.772 | 83.766 | 35.74 | 86.59 | [Link](https://huggingface.co/timm/deit3_base_patch16_224.fb_in1k) | fb_in1k |
| DeiT3_Base_Patch16_384 | (384,384,3) | 85.052 | 85.064 | 115.02 | 86.88 | [Link](https://huggingface.co/timm/deit3_base_patch16_384.fb_in1k) | fb_in1k |
| DeiT3_Large_Patch16_224 | (224,224,3) | 84.696 | 84.736 | 124.73 | 304.38 | [Link](https://huggingface.co/timm/deit3_large_patch16_224.fb_in1k) | fb_in1k |
| DeiT3_Large_Patch16_384 | (384,384,3) | 85.820 | 85.826 | 392.94 | 304.76 | [Link](https://huggingface.co/timm/deit3_large_patch16_384.fb_in1k) | fb_in1k |
| DenseNet121 | (224,224,3) | 74.186 | 74.422 | 6.37 | 8.04 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html) | |
| DenseNet161 | (224,224,3) | 77.200 | 77.142 | 16.85 | 28.86 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet161.html) | |
| DenseNet169 | (224,224,3) | 75.530 | 75.568 | 7.62 | 14.28 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet169.html) | |
| DenseNet201 | (224,224,3) | 76.760 | 76.882 | 9.82 | 20.21 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet201.html) | |
| FlexiViT_Small | (240,240,3) | 82.376 | 82.536 | 11.05 | 22.06 | [Link](https://huggingface.co/timm/flexivit_small.1200ep_in1k) | 1200ep_in1k |
| FlexiViT_Base | (240,240,3) | 84.726 | 84.664 | 41.30 | 86.59 | [Link](https://huggingface.co/timm/flexivit_base.1200ep_in1k) | 1200ep_in1k |
| FlexiViT_Large | (240,240,3) | 85.640 | 85.658 | 143.89 | 304.36 | [Link](https://huggingface.co/timm/flexivit_large.1200ep_in1k) | 1200ep_in1k |
| GoogLeNet | (224,224,3) | 69.850 | 69.778 | 3.04 | 6.62 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.googlenet.html) | |
| Inception_V3 | (299,299,3) | 77.222 | 77.278 | 11.51 | 23.82 | [Link](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html) | |
| MNASNet1_0 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mnasnet1_0.html) | |
| MNASNet1_3 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mnasnet1_3.html) | |
| MobileNet_V2 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mobilenet_v2.html) | IMAGENET1K_V2 |
| RegNet_X_400MF | (224,224,3) | 72.494 | 72.908 | 0.84 | 5.48 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_400mf.html) | IMAGENET1K_V1 |
| RegNet_X_400MF | (224,224,3) | 74.218 | 74.860 | 0.84 | 5.48 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_400mf.html) | IMAGENET1K_V2 |
| RegNet_X_800MF | (224,224,3) | 74.958 | 74.210 | 1.62 | 7.24 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_800mf.html) | IMAGENET1K_V1 |
| RegNet_X_800MF | (224,224,3) | 77.096 | 77.496 | 1.62 | 7.24 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_800mf.html) | IMAGENET1K_V2 |
| RegNet_X_1_6GF | (224,224,3) | 76.866 | 77.084 | 3.24 | 9.17 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_1_6gf.html) | IMAGENET1K_V1 |
| RegNet_X_1_6GF | (224,224,3) | 79.150 | 79.676 | 3.24 | 9.17 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_1_6gf.html) | IMAGENET1K_V2 |
| RegNet_X_3_2GF | (224,224,3) | 78.164 | 78.342 | 6.40 | 15.27 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html) | IMAGENET1K_V1 |
| RegNet_X_3_2GF | (224,224,3) | 80.832 | 81.194 | 6.40 | 15.27 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html) | IMAGENET1K_V2 |
| RegNet_X_8GF | (224,224,3) | 79.300 | 79.372 | 16.05 | 39.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_8gf.html) | IMAGENET1K_V1 |
| RegNet_X_8GF | (224,224,3) | 81.368 | 81.692 | 16.05 | 39.52 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_8gf.html) | IMAGENET1K_V2 |
| RegNet_X_16GF | (224,224,3) | 79.922 | 80.092 | 31.99 | 54.22 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_16gf.html) | IMAGENET1K_V1 |
| RegNet_X_16GF | (224,224,3) | 82.438 | 82.712 | 31.99 | 54.22 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_16gf.html) | IMAGENET1K_V2 |
| RegNet_X_32GF | (224,224,3) | 80.532 | 80.592 | 63.63 | 107.73 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_32gf.html) | IMAGENET1K_V1 |
| RegNet_X_32GF | (224,224,3) | 82.868 | 83.022 | 63.63 | 107.73 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_32gf.html) | IMAGENET1K_V2 |
| RegNet_Y_400MF | (224,224,3) | 73.672 | 74.004 | 0.82 | 4.33 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html) | IMAGENET1K_V1 |
| RegNet_Y_400MF | (224,224,3) | 75.306 | 75.802 | 0.82 | 4.33 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html) | IMAGENET1K_V2 |
| RegNet_Y_800MF | (224,224,3) | 76.188 | 76.396 | 1.70 | 6.42 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_800mf.html) | IMAGENET1K_V1 |
| RegNet_Y_800MF | (224,224,3) | 78.430 | 78.890 | 1.70 | 6.42 | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_800mf.html) | IMAGENET1K_V2 |
| RegNet_Y_1_6GF | (224,224,3) | 77.424 | 77.926 | 3.27 | 11.18 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_1_6gf.html) | IMAGENET1K_V1 |
| RegNet_Y_1_6GF | (224,224,3) | 80.496 | 80.882 | 3.27 | 11.18 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_1_6gf.html) | IMAGENET1K_V2 |
| RegNet_Y_3_2GF | (224,224,3) | 78.642 | 78.962 | 6.41 | 19.40 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_y_3_2gf.html) | IMAGENET1K_V1 |
| RegNet_Y_3_2GF | (224,224,3) | 81.514 | 82.018 | 6.41 | 19.40 | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_y_3_2gf.html) | IMAGENET1K_V2 |
| RegNet_Y_8GF | (224,224,3) | 79.856 | 80.052 | 17.05 | 39.34 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_8gf.html) | IMAGENET1K_V1 |
| RegNet_Y_8GF | (224,224,3) | 82.556 | 82.824 | 17.05 | 39.34 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_8gf.html) | IMAGENET1K_V2 |
| RegNet_Y_16GF | (224,224,3) | 80.278 | 80.424 | 31.95 | 83.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_16gf.html) | IMAGENET1K_V1 |
| RegNet_Y_16GF | (224,224,3) | 82.464 | 82.862 | 31.95 | 83.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_16gf.html) | IMAGENET1K_V2 |
| RegNet_Y_32GF | (224,224,3) | 80.696 | 80.834 | 64.72 | 144.97 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_32gf.html) | IMAGENET1K_V1 |
| RegNet_Y_32GF | (224,224,3) | 82.892 | 83.362 | 64.72 | 144.97 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_32gf.html) | IMAGENET1K_V2 |
| ResNet18 | (224,224,3) | 69.570 | 69.778 | 3.64 | 11.68 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet18.html) | |
| ResNet34 | (224,224,3) | 73.156 | 73.304 | 7.35 | 21.79 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet34.html) | |
| ResNet50 | (224,224,3) | 75.976 | 76.116 | 8.23 | 25.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet50.html) | IMAGENET1K_V1 |
| ResNet50 | (224,224,3) | 80.598 | 80.852 | 8.23 | 25.53 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet50.html) | IMAGENET1K_V2 |
| ResNet101 | (224,224,3) | 77.148 | 77.350 | 15.69 | 44.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet101.html) | IMAGENET1K_V1 |
| ResNet101 | (224,224,3) | 81.516 | 81.914 | 15.69 | 44.50 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet101.html) | IMAGENET1K_V2 |
| ResNet152 | (224,224,3) | 78.064 | 78.306 | 23.14 | 60.12 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet152.html) | IMAGENET1K_V1 |
| ResNet152 | (224,224,3) | 82.014 | 82.272 | 23.14 | 60.12 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet152.html) | IMAGENET1K_V2 |
| ResNeXt50_32X4D | (224,224,3) | 77.552 | 77.634 | 8.53 | 24.99 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext50_32x4d.html) | IMAGENET1K_V1 |
| ResNeXt50_32X4D | (224,224,3) | 80.858 | 81.212 | 8.53 | 24.99 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext50_32x4d.html) | IMAGENET1K_V2 |
| ResNeXt101_32X8D | (224,224,3) | 79.186 | 79.290 | 32.97 | 88.69 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_32x8d.html) | IMAGENET1K_V1 |
| ResNeXt101_32X8D | (224,224,3) | 82.616 | 82.784 | 32.97 | 88.69 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_32x8d.html) | IMAGENET1K_V2 |
| ResNeXt101_64X4D | (224,224,3) | 82.994 | 83.234 | 31.06 | 83.35 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_64x4d.html) | |
| ShuffleNet_V2_X1_0 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_0.html) | |
| ShuffleNet_V2_X1_5 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_5.html) | |
| ShuffleNet_V2_X2_0 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x2_0.html) | |
| VGG11 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg11.html) | |
| VGG11_BN | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg11_bn.html) | |
| VGG13 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg13.html) | |
| VGG13_BN | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg13_bn.html) | |
| VGG16 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg16.html) | |
| VGG16_BN | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg16_bn.html) | |
| VGG19 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg19.html) | |
| VGG19_BN | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg19_bn.html) | |
| ViT_Tiny_Patch16_224 | (224,224,3) | 75.414 | 75.456 | 2.66 | 5.72 | [Link](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Tiny_Patch16_384 | (384,384,3) | 78.376 | 78.470 | 10.37 | 5.79 | [Link](https://huggingface.co/timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Small_Patch16_224 | (224,224,3) | 81.434 | 81.412 | 9.50 | 22.05 | [Link](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Small_Patch16_384 | (384,384,3) | 83.774 | 83.782 | 33.00 | 22.20 | [Link](https://huggingface.co/timm/vit_base_patch16_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Small_Patch32_224 | (224,224,3) | 75.926 | 75.922 | 2.32 | 22.88 | [Link](https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Small_Patch32_384 | (384,384,3) | 80.450 | 80.452 | 7.07 | 22.92 | [Link](https://huggingface.co/timm/vit_base_patch32_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Base_Patch8_224 | (224,224,3) | 86.342 | 86.268 | 163.48 | 86.58 | [Link](https://huggingface.co/timm/vit_base_patch8_224.augreg2_in21k_ft_in1k) | augreg2_in21k_ft_in1k |
| ViT_Base_Patch16_224 | (224,224,3) | 85.118 | 85.112 | 35.73 | 86.57 | [Link](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k) | augreg2_in21k_ft_in1k |
| ViT_Base_Patch16_384 | (384,384,3) | 86.074 | 86.018 | 115.00 | 86.86 | [Link](https://huggingface.co/timm/vit_base_patch16_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Base_Patch32_224 | (224,224,3) | 80.672 | 80.698 | 8.89 | 88.22 | [Link](https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Base_Patch32_384 | (384,384,3) | 83.422 | 83.398 | 26.45 | 88.30 | [Link](https://huggingface.co/timm/vit_base_patch32_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Large_Patch16_224 | (224,224,3) | 85.920 | 85.870 | 124.71 | 304.33 | [Link](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Large_Patch16_384 | (384,384,3) | 87.076 | 87.086 | 392.88 | 304.72 | [Link](https://huggingface.co/timm/vit_large_patch16_384.augreg_in21k_ft_in1k) | augreg_in21k_ft_in1k |
| ViT_Large_Patch32_384 | (384,384,3) | 81.562 | 81.512 | 91.52 | 306.63 | [Link](https://huggingface.co/timm/vit_large_patch32_384.orig_in21k_ft_in1k) | orig_in21k_ft_in1k |
| Wide_ResNet50_2 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html) | IMAGENET1K_V1 |
| Wide_ResNet50_2 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html) | IMAGENET1K_V2 |
| Wide_ResNet101_2 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet101_2.html) | IMAGENET1K_V1 |
| Wide_ResNet101_2 | (224,224,3) |  |  |  |  | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet101_2.html) | IMAGENET1K_V2 |
| YOLOv5n-cls | (224,224,3) | 63.366 | 63.982 | 0.43 | 2.49 | [Link](https://github.com/ultralytics/yolov5/) | |
| YOLOv5s-cls | (224,224,3) | 70.564 | 70.854 | 1.42 | 5.45 | [Link](https://github.com/ultralytics/yolov5/) | |
| YOLOv5m-cls | (224,224,3) | 75.194 | 75.418 | 4.03 | 12.95 | [Link](https://github.com/ultralytics/yolov5/) | |
| YOLOv5l-cls | (224,224,3) | 77.226 | 77.528 | 8.82 | 26.54 | [Link](https://github.com/ultralytics/yolov5/) | |
| YOLOv5x-cls | (224,224,3) | 78.180 | 78.312 | 16.46 | 48.07 | [Link](https://github.com/ultralytics/yolov5/) | |
| YOLOv8s-cls | (224,224,3) | 72.352 | 73.774 | 1.67 | 6.36 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8m-cls | (224,224,3) | 76.038 | 76.824 | 5.37 | 17.04 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8l-cls | (224,224,3) | 77.660 | 78.276 | 12.53 | 37.47 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8x-cls | (224,224,3) | 78.544 | 78.936 | 19.38 | 57.40 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLO11s-cls | (224,224,3) | 74.594 | 75.244 | 1.63 | 6.72 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11m-cls | (224,224,3) | 76.662 | 77.388 | 5.17 | 11.62 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11l-cls | (224,224,3) | 77.676 | 78.284 | 6.51 | 14.10 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11x-cls | (224,224,3) | 78.808 | 79.426 | 14.20 | 29.61 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO26n-cls | (224,224,3) | 70.118 | 71.304 | 0.47 | 2.81 | [Link](https://docs.ultralytics.com/models/yolo26/) | |
| YOLO26s-cls | (224,224,3) | 75.138 | 75.910 | 1.63 | 6.72 | [Link](https://docs.ultralytics.com/models/yolo26/) | |
| YOLO26m-cls | (224,224,3) | 77.536 | 78.078 | 5.17 | 11.62 | [Link](https://docs.ultralytics.com/models/yolo26/) | |
| YOLO26l-cls | (224,224,3) | 78.576 | 79.060 | 6.51 | 14.10 | [Link](https://docs.ultralytics.com/models/yolo26/) | |
| YOLO26x-cls | (224,224,3) | 79.576 | 79.866 | 14.20 | 29.61 | [Link](https://docs.ultralytics.com/models/yolo26/) | |
<details>
<summary>Image Classification (ImageNet)</summary>

- Acc<sup>Top1</sup> values are model accuracies on the [ImageNet](https://www.image-net.org/index.php) dataset validation set.

</details>

## Object Detection

| Model | Input Size<br>(H,W,C) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{box}}}$<br>(NPU) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{box}}}$<br>(GPU) | FLOPs (B) | params (M) | Source | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv3 | (640,640,3) | 46.081 | 46.839 | 162.27 | 61.92 | [Link](https://docs.ultralytics.com/models/yolov3/) | |
| YOLOv3u | (640,640,3) | 51.178 | 51.582 | 289.23 | 103.77 | [Link](https://docs.ultralytics.com/models/yolov3/) | |
| YOLOv3-spp | (640,640,3) | 46.848 | 47.616 | 163.23 | 62.97 | [Link](https://github.com/ultralytics/yolov3/) | |
| YOLOv3-sppu | (640,640,3) | 51.441 | 51.754 | 290.20 | 104.82 | [Link](https://docs.ultralytics.com/models/yolov3/) | |
| YOLOv5su | (640,640,3) | 42.144 | 42.877 | 25.93 | 9.18 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5s6u | (1280,1280,3) | 48.046 | 48.636 | 105.38 | 15.46 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5m6 | (1280,1280,3) | 50.099 | 51.078 | 212.83 | 35.71 | [Link](https://github.com/ultralytics/yolov5/) | |
| YOLOv5mu | (640,640,3) | 48.344 | 48.910 | 67.70 | 25.13 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5m6u | (1280,1280,3) | 53.037 | 53.476 | 275.41 | 41.36 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5l6 | (1280,1280,3) | 52.667 | 53.376 | 466.00 | 76.73 | [Link](https://github.com/ultralytics/yolov5/) | |
| YOLOv5lu | (640,640,3) | 51.847 | 52.172 | 140.50 | 53.24 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5l6u | (1280,1280,3) | 55.136 | 55.466 | 571.74 | 86.22 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5x6 | (1280,1280,3) | 54.058 | 54.706 | 869.05 | 140.73 | [Link](https://github.com/ultralytics/yolov5/) | |
| YOLOv5xu | (640,640,3) | 52.823 | 53.090 | 254.38 | 97.27 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv5x6u | (1280,1280,3) | 56.156 | 56.488 | 1035.24 | 155.65 | [Link](https://docs.ultralytics.com/models/yolov5/) | |
| YOLOv7 | (640,640,3) | 50.430 | 50.942 | 110.55 | 36.91 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv7d6 | (1280,1280,3) | 55.572 | 55.799 | 730.75 | 133.76 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv7e6 | (1280,1280,3) | 55.399 | 55.567 | 538.42 | 97.20 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv7e6e | (1280,1280,3) | 56.011 | 56.298 | 878.43 | 151.69 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv7w6 | (1280,1280,3) | 53.989 | 54.128 | 374.79 | 70.39 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv7x | (640,640,3) | 52.302 | 52.706 | 197.67 | 71.31 | [Link](https://github.com/WongKinYiu/yolov7/) | |
| YOLOv8s | (640,640,3) | 44.254 | 44.918 | 30.49 | 11.20 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8m | (640,640,3) | 49.677 | 50.239 | 82.26 | 25.93 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8l | (640,640,3) | 52.367 | 52.772 | 170.26 | 43.71 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8x | (640,640,3) | 53.393 | 53.802 | 264.17 | 68.24 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| GELANs | (640,640,3) | 45.585 | 46.486 | 29.01 | 7.15 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9s | (640,640,3) | 45.879 | 46.777 | 29.01 | 7.11 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| GELANm | (640,640,3) | 50.385 | 50.928 | 80.81 | 20.02 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9m | (640,640,3) | 50.543 | 51.191 | 80.81 | 19.98 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| GELANc | (640,640,3) | 51.944 | 52.273 | 107.83 | 25.33 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9c | (640,640,3) | 52.388 | 52.917 | 107.83 | 25.29 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| GELANe | (640,640,3) | 54.416 | 52.927 | 199.86 | 57.39 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9e | (640,640,3) | 55.241 | 55.528 | 199.86 | 57.35 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv10s | (640,640,3) | 45.168 | 56.015 | 23.98 | 7.25 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLOv10m | (640,640,3) | 49.857 | 50.838 | 63.39 | 15.36 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLOv10b | (640,640,3) | 51.115 | 52.096 | 97.74 | 19.11 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLOv10l | (640,640,3) | 52.112 | 52.816 | 127.21 | 24.37 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLOv10x | (640,640,3) | 53.123 | 53.993 | 170.01 | 29.47 | [Link](https://docs.ultralytics.com/models/yolov10/) | |
| YOLO11s | (640,640,3) | 45.955 | 46.617 | 23.80 | 9.49 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11m | (640,640,3) | 50.480 | 51.310 | 72.81 | 20.13 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11l | (640,640,3) | 52.565 | 53.165 | 93.21 | 25.38 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11x | (640,640,3) | 54.073 | 54.478 | 204.31 | 56.96 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO12s | (640,640,3) | 46.910 | 47.687 | 26.73 | 9.30 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12m | (640,640,3) | 51.689 | 52.297 |  77.22| 20.21 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12l | (640,640,3) | 52.691 | 53.508 | 105.07 | 26.44 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12x | (640,640,3) | 54.304 | 55.061 |  223.27| 59.18 | [Link](https://docs.ultralytics.com/models/yolo12/) | |


<details>
<summary>Object Detection (COCO)</summary>

- $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{box}}}$ values are for single-model single-scale on the [COCO val2017](https://cocodataset.org/) dataset.

</details>

## Instance Segmentation

| Model | Input Size<br>(H,W,C) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{mask}}}$<br>(NPU) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{mask}}}$<br>(GPU) | FLOPs (B) | params (M) | Source | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5l-seg | (640,640,3) | 39.370 | 39.942 | 153.53 | 47.89 | [Link](https://github.com/ultralytics/yolov5/) | |
| YOLOv5x-seg | (640,640,3) | 40.977 | 41.318 | 273.99 | 88.77 | [Link](https://github.com/ultralytics/yolov5/) | |
| YOLOv8s-seg | (640,640,3) | 36.190 | 36.691 | 44.86 | 11.81 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8m-seg | (640,640,3) | 40.249 | 40.596 | 114.06 | 27.27 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8l-seg | (640,640,3) | 42.187 | 42.462 | 226.27 | 45.97 | [Link](https://docs.ultralytics.com/models/yolov8/) |  |
| YOLOv8x-seg | (640,640,3) | 42.998 | 43.155 | 351.31 | 71.80 | [Link](https://docs.ultralytics.com/models/yolov8/) |  |
| GELANc-seg | (640,640,3) | 41.894 | 42.204 | 150.93 | 27.47 | [Link](https://docs.ultralytics.com/models/yolov9/) | |
| YOLOv9c-seg | (640,640,3) | 42.246 | 42.655 | 152.44 | 27.45 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLOv9e-seg | (640,640,3) | 44.123 | 44.394 | 256.38 | 59.74 | [Link](https://github.com/WongKinYiu/yolov9/) | |
| YOLO11s-seg | (640,640,3) | 37.224 | 37.726 | 38.18 | 10.14 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11m-seg | (640,640,3) | 41.219 | 41.683 | 128.82 | 22.44 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11l-seg | (640,640,3) | 42.637 | 42.974 | 149.22 | 27.69 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11x-seg | (640,640,3) | 43.618 | 43.910 | 329.44 | 62.14 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO12m-seg | (640,640,3) |  |  |  |  | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12l-seg | (640,640,3) | 43.063 | 43.456 | 154.99 | 28.80 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO12x-seg | (640,640,3) | 43.839 | 44.418 | 334.55 | 64.55 | [Link](https://docs.ultralytics.com/models/yolo12/) | |
| YOLO26x-seg | (640,640,3) | 45.899 | 46.717 | 346.96 | 62.86 | [Link](https://docs.ultralytics.com/models/yolo26/) | |

<details>
<summary> Instance Segmentation (COCO)</summary>

- $\underset{50\text{–}95}{\text{mAP}_{\text{val}}^{\text{mask}}}$ values are for single-model single-scale on the [COCO val2017](https://cocodataset.org/) dataset.

</details>

## Pose Estimation

| Model | Input Size<br>(H,W,C) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{pose}}}$<br>(NPU) | $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{pose}}}$<br>(GPU) | FLOPs (B) | params (M) | Source | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8l-pose | (640,640,3) | 65.453 | 66.822 |  173.70| 44.53 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8x-pose | (640,640,3) | 69.525 | 70.758 | 269.63 | 69.53 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLOv8x-pose-p6 | (1280,1280,3) | 69.525 | 70.758 | 1092.51 | 99.41 | [Link](https://docs.ultralytics.com/models/yolov8/) | |
| YOLO11l-pose | (640,640,3) | 63.700 | 65.424 | 96.65 | 26.21 | [Link](https://docs.ultralytics.com/models/yolo11/) | |
| YOLO11x-pose | (640,640,3) | 67.430 | 68.600 |  212.26| 58.82 | [Link](https://docs.ultralytics.com/models/yolo11/) | |

<details>
<summary>Pose Estimation (COCO)</summary>

- $\underset{\texttt{50-95}}{\texttt{mAP}_{\texttt{val}}^{\texttt{pose}}}$ values are for single-model single-scale on the [COCO Keypoints val2017](https://cocodataset.org/) dataset.

</details>
