from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class MNasNet1_0_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/mnasnet1_0_IMAGENET1K_V1/aries/single/mnasnet1_0_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/mnasnet1_0_IMAGENET1K_V1/aries/multi/mnasnet1_0_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/mnasnet1_0_IMAGENET1K_V1/aries/global/mnasnet1_0_IMAGENET1K_V1.mxq",
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class MNasNet1_3_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/mnasnet1_3_IMAGENET1K_V1/aries/single/mnasnet1_3_IMAGENET1K_V1.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/mnasnet1_3_IMAGENET1K_V1/aries/multi/mnasnet1_3_IMAGENET1K_V1.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/mnasnet1_3_IMAGENET1K_V1/aries/global/mnasnet1_3_IMAGENET1K_V1.mxq",
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
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1  # Default model


def MNasNet1_0(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        MNasNet1_0_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def MNasNet1_3(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        MNasNet1_3_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
