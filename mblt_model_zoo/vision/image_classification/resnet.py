from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ResNet18_Set(ModelInfoSet):
    """ResNet18 model info set."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnet18_IMAGENET1K_V1/aries/single/resnet18_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnet18_IMAGENET1K_V1/aries/multi/resnet18_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnet18_IMAGENET1K_V1/aries/global/resnet18_IMAGENET1K_V1.mxq",
                    "global4": "https://dl.mobilint.com/model/vision/image_classification/resnet18_IMAGENET1K_V1/aries/global4/resnet18_IMAGENET1K_V1.mxq",
                    "global8": "https://dl.mobilint.com/model/vision/image_classification/resnet18_IMAGENET1K_V1/aries/global8/resnet18_IMAGENET1K_V1.mxq",
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 256,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class ResNet34_Set(ModelInfoSet):
    """ResNet34 model info set."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnet34_IMAGENET1K_V1/aries/single/resnet34_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnet34_IMAGENET1K_V1/aries/multi/resnet34_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnet34_IMAGENET1K_V1/aries/global/resnet34_IMAGENET1K_V1.mxq",
                    "global4": "https://dl.mobilint.com/model/vision/image_classification/resnet34_IMAGENET1K_V1/aries/global4/resnet34_IMAGENET1K_V1.mxq",
                    "global8": "https://dl.mobilint.com/model/vision/image_classification/resnet34_IMAGENET1K_V1/aries/global8/resnet34_IMAGENET1K_V1.mxq",
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 256,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class ResNet50_Set(ModelInfoSet):
    """ResNet50 model info set."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnet50_IMAGENET1K_V1/aries/single/resnet50_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnet50_IMAGENET1K_V1/aries/multi/resnet50_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnet50_IMAGENET1K_V1/aries/global/resnet50_IMAGENET1K_V1.mxq",
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 256,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnet50_IMAGENET1K_V2/aries/single/resnet50_IMAGENET1K_V2.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnet50_IMAGENET1K_V2/aries/multi/resnet50_IMAGENET1K_V2.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnet50_IMAGENET1K_V2/aries/global/resnet50_IMAGENET1K_V2.mxq",
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 232,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class ResNet101_Set(ModelInfoSet):
    """ResNet101 model info set."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnet101_IMAGENET1K_V1/aries/single/resnet101_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnet101_IMAGENET1K_V1/aries/multi/resnet101_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnet101_IMAGENET1K_V1/aries/global/resnet101_IMAGENET1K_V1.mxq",
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 256,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnet101_IMAGENET1K_V2/aries/single/resnet101_IMAGENET1K_V2.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnet101_IMAGENET1K_V2/aries/multi/resnet101_IMAGENET1K_V2.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnet101_IMAGENET1K_V2/aries/global/resnet101_IMAGENET1K_V2.mxq",
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 232,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class ResNet152_Set(ModelInfoSet):
    """ResNet152 model info set."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnet152_IMAGENET1K_V1/aries/single/resnet152_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnet152_IMAGENET1K_V1/aries/multi/resnet152_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnet152_IMAGENET1K_V1/aries/global/resnet152_IMAGENET1K_V1.mxq",
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 256,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/resnet152_IMAGENET1K_V2/aries/single/resnet152_IMAGENET1K_V2.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/resnet152_IMAGENET1K_V2/aries/multi/resnet152_IMAGENET1K_V2.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/resnet152_IMAGENET1K_V2/aries/global/resnet152_IMAGENET1K_V2.mxq",
                    "global4": None,
                    "global8": None,
                },
                "regulus": {"single": None},
            },
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 232,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "Normalize": {"style": "torch"},
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V2  # Default model


def ResNet18(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the ResNet18 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "IMAGENET1K_V1"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        ResNet18_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ResNet34(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the ResNet34 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "IMAGENET1K_V1"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        ResNet34_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ResNet50(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the ResNet50 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        ResNet50_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ResNet101(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the ResNet101 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        ResNet101_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ResNet152(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the ResNet152 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        ResNet152_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
