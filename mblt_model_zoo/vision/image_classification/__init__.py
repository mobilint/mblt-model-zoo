from .alexnet import AlexNet
from .densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from .vgg import VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN
from .mnasnet import MNasNet0_5, MNasNet0_75, MNasNet1_0
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .regnet import (
    Regnet_X_16GF,
    Regnet_X_1_6GF,
    Regnet_X_32GF,
    Regnet_X_3_2GF,
    Regnet_X_400MF,
    Regnet_X_800MF,
    Regnet_X_8GF,
)
from .resnext import ResNext50_32x4d, ResNext101_64x4d, ResNext101_32x8d
from .efficientnet import EfficientNet_B1
from .mobilenet_v1 import MobileNet_V1
from .mobilenet_v2 import MobileNet_V2
from .shufflenet_v2 import (
    ShuffleNet_V2_X1_0,
    ShuffleNet_V2_X1_5,
    ShuffleNet_V2_X2_0,
)
