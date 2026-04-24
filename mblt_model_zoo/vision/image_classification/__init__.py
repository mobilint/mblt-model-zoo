"""Image classification model exports."""

from .._compat import create_model_class

__all__ = [
    "AlexNet",
    "CAFormer_B36",
    "CAFormer_M36",
    "CAFormer_S18",
    "CAFormer_S36",
    "CoAtNet_0_RW_224",
    "CoAtNet_1_RW_224",
    "CoAtNet_2_RW_224",
    "ConvFormer_B36",
    "ConvFormer_M36",
    "ConvFormer_S18",
    "ConvFormer_S36",
    "ConvNeXt_Base",
    "ConvNeXt_Large",
    "ConvNeXt_Small",
    "ConvNeXt_Tiny",
    "DeiT_Base_Patch16_224",
    "DeiT_Base_Patch16_384",
    "DeiT_Small_Patch16_224",
    "DeiT_Tiny_Patch16_224",
    "DeiT3_Base_Patch16_224",
    "DeiT3_Base_Patch16_384",
    "DeiT3_Large_Patch16_224",
    "DeiT3_Large_Patch16_384",
    "DeiT3_Medium_Patch16_224",
    "DeiT3_Small_Patch16_224",
    "DeiT3_Small_Patch16_384",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "DenseNet201",
    "EfficientFormer_L1",
    "EfficientFormer_L3",
    "EfficientFormer_L7",
    "EfficientNet_B0",
    "EfficientNet_B1",
    "EfficientNet_B2",
    "EfficientNet_B3",
    "EfficientNet_B4",
    "EfficientNet_B5",
    "EfficientNet_B6",
    "EfficientNet_B7",
    "EfficientNet_V2_L",
    "EfficientNet_V2_M",
    "EfficientNet_V2_S",
    "FlexiViT_Base",
    "FlexiViT_Large",
    "FlexiViT_Small",
    "GoogLeNet",
    "Inception_V3",
    "LeViT_Conv_128",
    "LeViT_Conv_128S",
    "LeViT_Conv_192",
    "LeViT_Conv_256",
    "LeViT_Conv_384",
    "MNASNet0_5",
    "MNASNet0_75",
    "MNASNet1_0",
    "MNASNet1_3",
    "MobileNet_V2",
    "MobileNet_V3_Large",
    "MobileNet_V3_Small",
    "RegNet_X_1_6GF",
    "RegNet_X_3_2GF",
    "RegNet_X_8GF",
    "RegNet_X_16GF",
    "RegNet_X_32GF",
    "RegNet_X_400MF",
    "RegNet_X_800MF",
    "RegNet_Y_1_6GF",
    "RegNet_Y_3_2GF",
    "RegNet_Y_8GF",
    "RegNet_Y_16GF",
    "RegNet_Y_32GF",
    "RegNet_Y_400MF",
    "RegNet_Y_800MF",
    "RepViT_M0_9",
    "RepViT_M1",
    "RepViT_M1_0",
    "RepViT_M1_1",
    "RepViT_M1_5",
    "RepViT_M2",
    "RepViT_M2_3",
    "RepViT_M3",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNeXt50_32x4d",
    "ResNeXt101_32x8d",
    "ResNeXt101_64x4d",
    "ShuffleNet_V2_X1_0",
    "ShuffleNet_V2_X1_5",
    "ShuffleNet_V2_X2_0",
    "SqueezeNet1_0",
    "SqueezeNet1_1",
    "Swin_B",
    "Swin_S",
    "Swin_T",
    "VGG11",
    "VGG11_BN",
    "VGG13",
    "VGG13_BN",
    "VGG16",
    "VGG16_BN",
    "VGG19",
    "VGG19_BN",
    "VisFormer_Small",
    "VisFormer_Tiny",
    "ViT_B_16",
    "ViT_B_32",
    "ViT_Base_Patch8_224",
    "ViT_Base_Patch16_224",
    "ViT_Base_Patch16_384",
    "ViT_Base_Patch32_224",
    "ViT_Base_Patch32_384",
    "ViT_L_16",
    "ViT_L_32",
    "ViT_Large_Patch16_224",
    "ViT_Large_Patch16_384",
    "ViT_Large_Patch32_384",
    "ViT_Small_Patch16_224",
    "ViT_Small_Patch16_384",
    "ViT_Small_Patch32_224",
    "ViT_Small_Patch32_384",
    "ViT_Tiny_Patch16_224",
    "ViT_Tiny_Patch16_384",
    "Wide_ResNet50_2",
    "Wide_ResNet101_2",
    "YOLO11lCls",
    "YOLO11mCls",
    "YOLO11nCls",
    "YOLO11sCls",
    "YOLO11xCls",
    "YOLO26lCls",
    "YOLO26mCls",
    "YOLO26nCls",
    "YOLO26sCls",
    "YOLO26xCls",
    "YOLOv5lCls",
    "YOLOv5mCls",
    "YOLOv5nCls",
    "YOLOv5sCls",
    "YOLOv5xCls",
    "YOLOv8lCls",
    "YOLOv8mCls",
    "YOLOv8nCls",
    "YOLOv8sCls",
    "YOLOv8xCls",
]

AlexNet = create_model_class("AlexNet", __name__)
CAFormer_B36 = create_model_class("CAFormer_B36", __name__)
CAFormer_M36 = create_model_class("CAFormer_M36", __name__)
CAFormer_S18 = create_model_class("CAFormer_S18", __name__)
CAFormer_S36 = create_model_class("CAFormer_S36", __name__)
CoAtNet_0_RW_224 = create_model_class("CoAtNet_0_RW_224", __name__)
CoAtNet_1_RW_224 = create_model_class("CoAtNet_1_RW_224", __name__)
CoAtNet_2_RW_224 = create_model_class("CoAtNet_2_RW_224", __name__)
ConvFormer_B36 = create_model_class("ConvFormer_B36", __name__)
ConvFormer_M36 = create_model_class("ConvFormer_M36", __name__)
ConvFormer_S18 = create_model_class("ConvFormer_S18", __name__)
ConvFormer_S36 = create_model_class("ConvFormer_S36", __name__)
ConvNeXt_Base = create_model_class("ConvNeXt_Base", __name__)
ConvNeXt_Large = create_model_class("ConvNeXt_Large", __name__)
ConvNeXt_Small = create_model_class("ConvNeXt_Small", __name__)
ConvNeXt_Tiny = create_model_class("ConvNeXt_Tiny", __name__)
DeiT_Base_Patch16_224 = create_model_class("DeiT_Base_Patch16_224", __name__)
DeiT_Base_Patch16_384 = create_model_class("DeiT_Base_Patch16_384", __name__)
DeiT_Small_Patch16_224 = create_model_class("DeiT_Small_Patch16_224", __name__)
DeiT_Tiny_Patch16_224 = create_model_class("DeiT_Tiny_Patch16_224", __name__)
DeiT3_Base_Patch16_224 = create_model_class("DeiT3_Base_Patch16_224", __name__)
DeiT3_Base_Patch16_384 = create_model_class("DeiT3_Base_Patch16_384", __name__)
DeiT3_Large_Patch16_224 = create_model_class("DeiT3_Large_Patch16_224", __name__)
DeiT3_Large_Patch16_384 = create_model_class("DeiT3_Large_Patch16_384", __name__)
DeiT3_Medium_Patch16_224 = create_model_class("DeiT3_Medium_Patch16_224", __name__)
DeiT3_Small_Patch16_224 = create_model_class("DeiT3_Small_Patch16_224", __name__)
DeiT3_Small_Patch16_384 = create_model_class("DeiT3_Small_Patch16_384", __name__)
DenseNet121 = create_model_class("DenseNet121", __name__)
DenseNet161 = create_model_class("DenseNet161", __name__)
DenseNet169 = create_model_class("DenseNet169", __name__)
DenseNet201 = create_model_class("DenseNet201", __name__)
EfficientFormer_L1 = create_model_class("EfficientFormer_L1", __name__)
EfficientFormer_L3 = create_model_class("EfficientFormer_L3", __name__)
EfficientFormer_L7 = create_model_class("EfficientFormer_L7", __name__)
EfficientNet_B0 = create_model_class("EfficientNet_B0", __name__)
EfficientNet_B1 = create_model_class("EfficientNet_B1", __name__)
EfficientNet_B2 = create_model_class("EfficientNet_B2", __name__)
EfficientNet_B3 = create_model_class("EfficientNet_B3", __name__)
EfficientNet_B4 = create_model_class("EfficientNet_B4", __name__)
EfficientNet_B5 = create_model_class("EfficientNet_B5", __name__)
EfficientNet_B6 = create_model_class("EfficientNet_B6", __name__)
EfficientNet_B7 = create_model_class("EfficientNet_B7", __name__)
EfficientNet_V2_L = create_model_class("EfficientNet_V2_L", __name__)
EfficientNet_V2_M = create_model_class("EfficientNet_V2_M", __name__)
EfficientNet_V2_S = create_model_class("EfficientNet_V2_S", __name__)
FlexiViT_Base = create_model_class("FlexiViT_Base", __name__)
FlexiViT_Large = create_model_class("FlexiViT_Large", __name__)
FlexiViT_Small = create_model_class("FlexiViT_Small", __name__)
GoogLeNet = create_model_class("GoogLeNet", __name__)
Inception_V3 = create_model_class("Inception_V3", __name__)
LeViT_Conv_128 = create_model_class("LeViT_Conv_128", __name__)
LeViT_Conv_128S = create_model_class("LeViT_Conv_128S", __name__)
LeViT_Conv_192 = create_model_class("LeViT_Conv_192", __name__)
LeViT_Conv_256 = create_model_class("LeViT_Conv_256", __name__)
LeViT_Conv_384 = create_model_class("LeViT_Conv_384", __name__)
MNASNet0_5 = create_model_class("MNASNet0_5", __name__)
MNASNet0_75 = create_model_class("MNASNet0_75", __name__)
MNASNet1_0 = create_model_class("MNASNet1_0", __name__)
MNASNet1_3 = create_model_class("MNASNet1_3", __name__)
MobileNet_V2 = create_model_class("MobileNet_V2", __name__)
MobileNet_V3_Large = create_model_class("MobileNet_V3_Large", __name__)
MobileNet_V3_Small = create_model_class("MobileNet_V3_Small", __name__)
RegNet_X_1_6GF = create_model_class("RegNet_X_1_6GF", __name__)
RegNet_X_3_2GF = create_model_class("RegNet_X_3_2GF", __name__)
RegNet_X_8GF = create_model_class("RegNet_X_8GF", __name__)
RegNet_X_16GF = create_model_class("RegNet_X_16GF", __name__)
RegNet_X_32GF = create_model_class("RegNet_X_32GF", __name__)
RegNet_X_400MF = create_model_class("RegNet_X_400MF", __name__)
RegNet_X_800MF = create_model_class("RegNet_X_800MF", __name__)
RegNet_Y_1_6GF = create_model_class("RegNet_Y_1_6GF", __name__)
RegNet_Y_3_2GF = create_model_class("RegNet_Y_3_2GF", __name__)
RegNet_Y_8GF = create_model_class("RegNet_Y_8GF", __name__)
RegNet_Y_16GF = create_model_class("RegNet_Y_16GF", __name__)
RegNet_Y_32GF = create_model_class("RegNet_Y_32GF", __name__)
RegNet_Y_400MF = create_model_class("RegNet_Y_400MF", __name__)
RegNet_Y_800MF = create_model_class("RegNet_Y_800MF", __name__)
RepViT_M0_9 = create_model_class("RepViT_M0_9", __name__)
RepViT_M1 = create_model_class("RepViT_M1", __name__)
RepViT_M1_0 = create_model_class("RepViT_M1_0", __name__)
RepViT_M1_1 = create_model_class("RepViT_M1_1", __name__)
RepViT_M1_5 = create_model_class("RepViT_M1_5", __name__)
RepViT_M2 = create_model_class("RepViT_M2", __name__)
RepViT_M2_3 = create_model_class("RepViT_M2_3", __name__)
RepViT_M3 = create_model_class("RepViT_M3", __name__)
ResNet18 = create_model_class("ResNet18", __name__)
ResNet34 = create_model_class("ResNet34", __name__)
ResNet50 = create_model_class("ResNet50", __name__)
ResNet101 = create_model_class("ResNet101", __name__)
ResNet152 = create_model_class("ResNet152", __name__)
ResNeXt50_32x4d = create_model_class("ResNeXt50_32x4d", __name__)
ResNeXt101_32x8d = create_model_class("ResNeXt101_32x8d", __name__)
ResNeXt101_64x4d = create_model_class("ResNeXt101_64x4d", __name__)
ShuffleNet_V2_X1_0 = create_model_class("ShuffleNet_V2_X1_0", __name__)
ShuffleNet_V2_X1_5 = create_model_class("ShuffleNet_V2_X1_5", __name__)
ShuffleNet_V2_X2_0 = create_model_class("ShuffleNet_V2_X2_0", __name__)
SqueezeNet1_0 = create_model_class("SqueezeNet1_0", __name__)
SqueezeNet1_1 = create_model_class("SqueezeNet1_1", __name__)
Swin_B = create_model_class("Swin_B", __name__)
Swin_S = create_model_class("Swin_S", __name__)
Swin_T = create_model_class("Swin_T", __name__)
VGG11 = create_model_class("VGG11", __name__)
VGG11_BN = create_model_class("VGG11_BN", __name__)
VGG13 = create_model_class("VGG13", __name__)
VGG13_BN = create_model_class("VGG13_BN", __name__)
VGG16 = create_model_class("VGG16", __name__)
VGG16_BN = create_model_class("VGG16_BN", __name__)
VGG19 = create_model_class("VGG19", __name__)
VGG19_BN = create_model_class("VGG19_BN", __name__)
VisFormer_Small = create_model_class("VisFormer_Small", __name__)
VisFormer_Tiny = create_model_class("VisFormer_Tiny", __name__)
ViT_B_16 = create_model_class("ViT_B_16", __name__)
ViT_B_32 = create_model_class("ViT_B_32", __name__)
ViT_Base_Patch8_224 = create_model_class("ViT_Base_Patch8_224", __name__)
ViT_Base_Patch16_224 = create_model_class("ViT_Base_Patch16_224", __name__)
ViT_Base_Patch16_384 = create_model_class("ViT_Base_Patch16_384", __name__)
ViT_Base_Patch32_224 = create_model_class("ViT_Base_Patch32_224", __name__)
ViT_Base_Patch32_384 = create_model_class("ViT_Base_Patch32_384", __name__)
ViT_L_16 = create_model_class("ViT_L_16", __name__)
ViT_L_32 = create_model_class("ViT_L_32", __name__)
ViT_Large_Patch16_224 = create_model_class("ViT_Large_Patch16_224", __name__)
ViT_Large_Patch16_384 = create_model_class("ViT_Large_Patch16_384", __name__)
ViT_Large_Patch32_384 = create_model_class("ViT_Large_Patch32_384", __name__)
ViT_Small_Patch16_224 = create_model_class("ViT_Small_Patch16_224", __name__)
ViT_Small_Patch16_384 = create_model_class("ViT_Small_Patch16_384", __name__)
ViT_Small_Patch32_224 = create_model_class("ViT_Small_Patch32_224", __name__)
ViT_Small_Patch32_384 = create_model_class("ViT_Small_Patch32_384", __name__)
ViT_Tiny_Patch16_224 = create_model_class("ViT_Tiny_Patch16_224", __name__)
ViT_Tiny_Patch16_384 = create_model_class("ViT_Tiny_Patch16_384", __name__)
Wide_ResNet50_2 = create_model_class("Wide_ResNet50_2", __name__)
Wide_ResNet101_2 = create_model_class("Wide_ResNet101_2", __name__)
YOLO11lCls = create_model_class("YOLO11lCls", __name__)
YOLO11mCls = create_model_class("YOLO11mCls", __name__)
YOLO11nCls = create_model_class("YOLO11nCls", __name__)
YOLO11sCls = create_model_class("YOLO11sCls", __name__)
YOLO11xCls = create_model_class("YOLO11xCls", __name__)
YOLO26lCls = create_model_class("YOLO26lCls", __name__)
YOLO26mCls = create_model_class("YOLO26mCls", __name__)
YOLO26nCls = create_model_class("YOLO26nCls", __name__)
YOLO26sCls = create_model_class("YOLO26sCls", __name__)
YOLO26xCls = create_model_class("YOLO26xCls", __name__)
YOLOv5lCls = create_model_class("YOLOv5lCls", __name__)
YOLOv5mCls = create_model_class("YOLOv5mCls", __name__)
YOLOv5nCls = create_model_class("YOLOv5nCls", __name__)
YOLOv5sCls = create_model_class("YOLOv5sCls", __name__)
YOLOv5xCls = create_model_class("YOLOv5xCls", __name__)
YOLOv8lCls = create_model_class("YOLOv8lCls", __name__)
YOLOv8mCls = create_model_class("YOLOv8mCls", __name__)
YOLOv8nCls = create_model_class("YOLOv8nCls", __name__)
YOLOv8sCls = create_model_class("YOLOv8sCls", __name__)
YOLOv8xCls = create_model_class("YOLOv8xCls", __name__)
