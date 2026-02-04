"""
ResNeXt model definitions.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ResNext50_32x4d_Set(ModelInfoSet):
    """Configuration set for ResNeXt50 32x4d models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/resnext50_32x4d",
            "filename": "resnext50_32x4d.mxq",
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/resnext50_32x4d",
            "filename": "resnext50_32x4d.mxq",
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class ResNext101_32x8d_Set(ModelInfoSet):
    """Configuration set for ResNeXt101 32x8d models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/resnext101_32x8d",
            "filename": "resnext101_32x8d.mxq",
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/resnext101_32x8d",
            "filename": "resnext101_32x8d.mxq",
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class ResNext101_64x4d_Set(ModelInfoSet):
    """Configuration set for ResNeXt101 64x4d models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/resnext101_64x4d",
            "filename": "resnext101_64x4d.mxq",
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
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "image_classification",
        },
    )
    DEFAULT = IMAGENET1K_V1  # Default model


def ResNext50_32x4d(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a ResNext50_32x4d model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        ResNext50_32x4d_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ResNext101_32x8d(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a ResNext101_32x8d model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        ResNext101_32x8d_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def ResNext101_64x4d(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a ResNext101_64x4d model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        ResNext101_64x4d_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
