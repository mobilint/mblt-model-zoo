"""
DeiT3 (Data-efficient Image Transformers v3) model definitions.
"""

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class DeiT3_Small_Patch16_224_Set(ModelInfoSet):
    """Configuration set for DeiT3 Small Patch16 224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT3_Small_Patch16_224",
            "filename": "deit3_small_patch16_224.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT3_Small_Patch16_384_Set(ModelInfoSet):
    """Configuration set for DeiT3 Small Patch16 384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT3_Small_Patch16_384",
            "filename": "deit3_small_patch16_384.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 384,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT3_Medium_Patch16_224_Set(ModelInfoSet):
    """Configuration set for DeiT3 Medium Patch16 224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT3_Medium_Patch16_224",
            "filename": "deit3_medium_patch16_224.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT3_Base_Patch16_224_Set(ModelInfoSet):
    """Configuration set for DeiT3 Base Patch16 224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT3_Base_Patch16_224",
            "filename": "deit3_base_patch16_224.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT3_Base_Patch16_384_Set(ModelInfoSet):
    """Configuration set for DeiT3 Base Patch16 384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT3_Base_Patch16_384",
            "filename": "deit3_base_patch16_384.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 384,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT3_Large_Patch16_224_Set(ModelInfoSet):
    """Configuration set for DeiT3 Large Patch16 224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT3_Large_Patch16_224",
            "filename": "deit3_large_patch16_224.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 248,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


class DeiT3_Large_Patch16_384_Set(ModelInfoSet):
    """Configuration set for DeiT3 Large Patch16 384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT3_Large_Patch16_384",
            "filename": "deit3_large_patch16_384.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 384,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [384, 384],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )


def DeiT3_Small_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a DeiT3_Small_Patch16_224 model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        DeiT3_Small_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def DeiT3_Small_Patch16_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a DeiT3_Small_Patch16_384 model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        DeiT3_Small_Patch16_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def DeiT3_Medium_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a DeiT3_Medium_Patch16_224 model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        DeiT3_Medium_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def DeiT3_Base_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a DeiT3_Base_Patch16_224 model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        DeiT3_Base_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def DeiT3_Base_Patch16_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a DeiT3_Base_Patch16_384 model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        DeiT3_Base_Patch16_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def DeiT3_Large_Patch16_224(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a DeiT3_Large_Patch16_224 model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        DeiT3_Large_Patch16_224_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )


def DeiT3_Large_Patch16_384(
    local_path: str = None,
    model_type: str = "DEFAULT",
    infer_mode: str = "global",
    product: str = "aries",
) -> MBLT_Engine:
    """Constructs a DeiT3_Large_Patch16_384 model engine.

    Args:
        local_path (str, optional): Path to a local model file. Defaults to None.
        model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
        infer_mode (str, optional): Inference execution mode. Defaults to "global".
        product (str, optional): Target hardware product. Defaults to "aries".

    Returns:
        MBLT_Engine: A model engine instance.
    """
    return MBLT_Engine.from_model_info_set(
        DeiT3_Large_Patch16_384_Set,
        local_path=local_path,
        model_type=model_type,
        infer_mode=infer_mode,
        product=product,
    )
