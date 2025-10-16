from .alexnet import AlexNet
from .convnext import ConvNeXt_Tiny, ConvNeXt_Small, ConvNeXt_Base, ConvNeXt_Large
from .deit import (
    DeiT_Tiny_Patch16_224,
    DeiT_Small_Patch16_224,
    DeiT_Base_Patch16_224,
    DeiT_Base_Patch16_384,
)
from .deit3 import (
    DeiT3_Small_Patch16_224,
    DeiT3_Small_Patch16_384,
    DeiT3_Medium_Patch16_224,
    DeiT3_Base_Patch16_224,
    DeiT3_Base_Patch16_384,
    DeiT3_Large_Patch16_224,
    DeiT3_Large_Patch16_384,
)
from .densenet import DenseNet121, DenseNet169, DenseNet201
from .flexivit import FlexiViT_Small, FlexiViT_Base, FlexiViT_Large
from .googlenet import GoogLeNet
from .inception_v3 import Inception_V3
from .mnasnet import MNasNet1_0, MNasNet1_3
from .mobilenet_v2 import MobileNet_V2
from .regnet import (
    RegNet_X_16GF,
    RegNet_X_1_6GF,
    RegNet_X_32GF,
    RegNet_X_3_2GF,
    RegNet_X_400MF,
    RegNet_X_800MF,
    RegNet_X_8GF,
    RegNet_Y_16GF,
    RegNet_Y_1_6GF,
    RegNet_Y_32GF,
    RegNet_Y_3_2GF,
    RegNet_Y_400MF,
    RegNet_Y_800MF,
    RegNet_Y_8GF,
)
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .resnext import ResNext50_32x4d, ResNext101_64x4d, ResNext101_32x8d
from .shufflenet_v2 import (
    ShuffleNet_V2_X1_0,
    ShuffleNet_V2_X1_5,
    ShuffleNet_V2_X2_0,
)
from .vgg import VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN
from .vit import (
    ViT_Tiny_Patch16_224,
    ViT_Tiny_Patch16_384,
    ViT_Small_Patch16_224,
    ViT_Small_Patch16_384,
    ViT_Small_Patch32_224,
    ViT_Small_Patch32_384,
    ViT_Base_Patch8_224,
    ViT_Base_Patch16_224,
    ViT_Base_Patch16_384,
    ViT_Base_Patch32_224,
    ViT_Base_Patch32_384,
    ViT_Large_Patch16_224,
    ViT_Large_Patch16_384,
    ViT_Large_Patch32_384,
)
from .wide_resnet import Wide_ResNet50_2, Wide_ResNet101_2
