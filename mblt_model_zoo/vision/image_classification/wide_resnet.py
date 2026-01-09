from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class Wide_ResNet50_2_Set(ModelInfoSet):
    """Wide_ResNet50_2 model for image classification."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet50_2_IMAGENET1K_V1/aries/single/wide_resnet50_2_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet50_2_IMAGENET1K_V1/aries/multi/wide_resnet50_2_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet50_2_IMAGENET1K_V1/aries/global/wide_resnet50_2_IMAGENET1K_V1.mxq",
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
                    "single": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet50_2_IMAGENET1K_V2/aries/single/wide_resnet50_2_IMAGENET1K_V2.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet50_2_IMAGENET1K_V2/aries/multi/wide_resnet50_2_IMAGENET1K_V2.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet50_2_IMAGENET1K_V2/aries/global/wide_resnet50_2_IMAGENET1K_V2.mxq",
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


class Wide_ResNet101_2_Set(ModelInfoSet):
    """Wide_ResNet101_2 model for image classification."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet101_2_IMAGENET1K_V1/aries/single/wide_resnet101_2_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet101_2_IMAGENET1K_V1/aries/multi/wide_resnet101_2_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet101_2_IMAGENET1K_V1/aries/global/wide_resnet101_2_IMAGENET1K_V1.mxq",
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
                    "single": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet101_2_IMAGENET1K_V2/aries/single/wide_resnet101_2_IMAGENET1K_V2.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet101_2_IMAGENET1K_V2/aries/multi/wide_resnet101_2_IMAGENET1K_V2.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/wide_resnet101_2_IMAGENET1K_V2/aries/global/wide_resnet101_2_IMAGENET1K_V2.mxq",
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


def Wide_ResNet50_2(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the Wide_ResNet50_2 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        Wide_ResNet50_2_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def Wide_ResNet101_2(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """
    Load the Wide_ResNet101_2 model for the specified product and inference mode.

    Args:
        local_path (str, optional): Path to the local model file. Defaults to None.
        model_type (str, optional): Generic model type (e.g., "DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"). Defaults to "DEFAULT".
        infer_mode (str, optional): Inference mode ('single', 'multi', 'global', 'global4', 'global8'). Defaults to "global".
        product (str, optional): Target product ('aries', 'regulus'). Defaults to "aries".

    Returns:
        MBLT_Engine: An instance of the MBLT Engine configured for the specified model.
    """
    return MBLT_Engine.from_model_info_set(
        Wide_ResNet101_2_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
