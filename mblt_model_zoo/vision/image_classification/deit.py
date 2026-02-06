"""
DeiT (Data-efficient Image Transformers) model definitions.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class DeiT_Tiny_Patch16_224_Set(ModelInfoSet):
    """Configuration set for DeiT Tiny Patch16 224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT_Tiny_Patch16_224",
            "filename": "deit_tiny_patch16_224.mxq",
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


class DeiT_Small_Patch16_224_Set(ModelInfoSet):
    """Configuration set for DeiT Small Patch16 224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT_Small_Patch16_224",
            "filename": "deit_small_patch16_224.mxq",
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


class DeiT_Base_Patch16_224_Set(ModelInfoSet):
    """Configuration set for DeiT Base Patch16 224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT_Base_Patch16_224",
            "filename": "deit_base_patch16_224.mxq",
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


class DeiT_Base_Patch16_384_Set(ModelInfoSet):
    """Configuration set for DeiT Base Patch16 384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/DeiT_Base_Patch16_384",
            "filename": "deit_base_patch16_384.mxq",
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


class DeiT_Tiny_Patch16_224(MBLT_Engine):
    """DeiT_Tiny_Patch16_224 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        """Initializes the DeiT_Tiny_Patch16_224 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            DeiT_Tiny_Patch16_224_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class DeiT_Small_Patch16_224(MBLT_Engine):
    """DeiT_Small_Patch16_224 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        """Initializes the DeiT_Small_Patch16_224 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            DeiT_Small_Patch16_224_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class DeiT_Base_Patch16_224(MBLT_Engine):
    """DeiT_Base_Patch16_224 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        """Initializes the DeiT_Base_Patch16_224 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            DeiT_Base_Patch16_224_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class DeiT_Base_Patch16_384(MBLT_Engine):
    """DeiT_Base_Patch16_384 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        """Initializes the DeiT_Base_Patch16_384 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            DeiT_Base_Patch16_384_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
