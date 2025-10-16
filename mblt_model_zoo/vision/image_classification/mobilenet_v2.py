from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class MobileNet_V2_Set(ModelInfoSet):
    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": None,
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": "https://dl.mobilint.com/model/vision/image_classification/mobilenet_v2_IMAGENET1K_V2/aries/single/mobilenet_v2_IMAGENET1K_V2.mxq",
                    "multi": "https://dl.mobilint.com/model/vision/image_classification/mobilenet_v2_IMAGENET1K_V2/aries/multi/mobilenet_v2_IMAGENET1K_V2.mxq",
                    "global": "https://dl.mobilint.com/model/vision/image_classification/mobilenet_v2_IMAGENET1K_V2/aries/global/mobilenet_v2_IMAGENET1K_V2.mxq",
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
    DEFAULT = IMAGENET1K_V2  # Default model


def MobileNet_V2(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    return MBLT_Engine.from_model_info_set(
        MobileNet_V2_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
