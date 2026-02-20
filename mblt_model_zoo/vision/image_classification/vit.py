"""
Vision Transformer (ViT) model definitions.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ViT_B_16_Set(ModelInfoSet):
    """Configuration set for ViT_B_16 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_B_16.tv1_in1k",
            "filename": "vit_b_16_IMAGENET1K_V1.mxq",
            "revision": "main",
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
        post_cfg={"task": "image_classification"},
    )
    IMAGENET1K_SWAG_E2E_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_B_16.swag_e2e_in1k",
            "filename": "vit_b_16_IMAGENET1K_SWAG_E2E_V1.mxq",
            "revision": "main",
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
    IMAGENET1K_SWAG_LINEAR_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_B_16.swag_linear_in1k",
            "filename": "vit_b_16_IMAGENET1K_SWAG_LINEAR_V1.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 224,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1


class ViT_B_32_Set(ModelInfoSet):
    """Configuration set for ViT_B_32 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_B_32",
            "filename": "vit_b_32_IMAGENET1K_V1.mxq",
            "revision": "main",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_16_Set(ModelInfoSet):
    """Configuration set for ViT_L_16 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_L_16",
            "filename": "vit_l_16_IMAGENET1K_V1.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 242,
                "interpolation": "bilinear",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )
    IMAGENET1K_SWAG_E2E_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_L_16.swag_e2e_in1k",
            "filename": "vit_l_16_IMAGENET1K_SWAG_E2E_V1.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 512,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [512, 512],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )
    IMAGENET1K_SWAG_LINEAR_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_L_16.swag_linear_in1k",
            "filename": "vit_l_16_IMAGENET1K_SWAG_LINEAR_V1.mxq",
            "revision": "main",
        },
        pre_cfg={
            "Reader": {
                "style": "pil",
            },
            "Resize": {
                "size": 224,
                "interpolation": "bicubic",
            },
            "CenterCrop": {
                "size": [224, 224],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_32_Set(ModelInfoSet):
    """Configuration set for ViT_L_32 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_L_32",
            "filename": "vit_l_32_IMAGENET1K_V1.mxq",
            "revision": "main",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V1


class ViT_Tiny_Patch16_224_Set(ModelInfoSet):
    """Configuration set for ViT_Tiny_Patch16_224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Tiny_Patch16_224",
            "filename": "vit_tiny_patch16_224.mxq",
            "revision": "main",
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


class ViT_Tiny_Patch16_384_Set(ModelInfoSet):
    """Configuration set for ViT_Tiny_Patch16_384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Tiny_Patch16_384",
            "filename": "vit_tiny_patch16_384.mxq",
            "revision": "main",
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


class ViT_Small_Patch16_224_Set(ModelInfoSet):
    """Configuration set for ViT_Small_Patch16_224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Small_Patch16_224",
            "filename": "vit_small_patch16_224.mxq",
            "revision": "main",
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


class ViT_Small_Patch16_384_Set(ModelInfoSet):
    """Configuration set for ViT_Small_Patch16_384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Small_Patch16_384",
            "filename": "vit_small_patch16_384.mxq",
            "revision": "main",
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


class ViT_Small_Patch32_224_Set(ModelInfoSet):
    """Configuration set for ViT_Small_Patch32_224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Small_Patch32_224",
            "filename": "vit_small_patch32_224.mxq",
            "revision": "main",
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


class ViT_Small_Patch32_384_Set(ModelInfoSet):
    """Configuration set for ViT_Small_Patch32_384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Small_Patch32_384",
            "filename": "vit_small_patch32_384.mxq",
            "revision": "main",
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


class ViT_Base_Patch8_224_Set(ModelInfoSet):
    """Configuration set for ViT_Base_Patch8_224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Base_Patch8_224",
            "filename": "vit_base_patch8_224.mxq",
            "revision": "main",
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


class ViT_Base_Patch16_224_Set(ModelInfoSet):
    """Configuration set for ViT_Base_Patch16_224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Base_Patch16_224",
            "filename": "vit_base_patch16_224.mxq",
            "revision": "main",
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


class ViT_Base_Patch16_384_Set(ModelInfoSet):
    """Configuration set for ViT_Base_Patch16_384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Base_Patch16_384",
            "filename": "vit_base_patch16_384.mxq",
            "revision": "main",
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


class ViT_Base_Patch32_224_Set(ModelInfoSet):
    """Configuration set for ViT_Base_Patch32_224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Base_Patch32_224",
            "filename": "vit_base_patch32_224.mxq",
            "revision": "main",
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


class ViT_Base_Patch32_384_Set(ModelInfoSet):
    """Configuration set for ViT_Base_Patch32_384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Base_Patch32_384",
            "filename": "vit_base_patch32_384.mxq",
            "revision": "main",
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


class ViT_Large_Patch16_224_Set(ModelInfoSet):
    """Configuration set for ViT_Large_Patch16_224 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Large_Patch16_224",
            "filename": "vit_large_patch16_224.mxq",
            "revision": "main",
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


class ViT_Large_Patch16_384_Set(ModelInfoSet):
    """Configuration set for ViT_Large_Patch16_384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Large_Patch16_384",
            "filename": "vit_large_patch16_384.mxq",
            "revision": "main",
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


class ViT_Large_Patch32_384_Set(ModelInfoSet):
    """Configuration set for ViT_Large_Patch32_384 models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/ViT_Large_Patch32_384",
            "filename": "vit_large_patch32_384.mxq",
            "revision": "main",
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


class ViT_B_16(MBLT_Engine):
    """ViT_B_16 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_B_16 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_B_16_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_B_32(MBLT_Engine):
    """ViT_B_32 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_B_32 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_B_32_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_L_16(MBLT_Engine):
    """ViT_L_16 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_L_16 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_L_16_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_L_32(MBLT_Engine):
    """ViT_L_32 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_L_32 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_L_32_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Tiny_Patch16_224(MBLT_Engine):
    """ViT_Tiny_Patch16_224 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Tiny_Patch16_224 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Tiny_Patch16_224_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Tiny_Patch16_384(MBLT_Engine):
    """ViT_Tiny_Patch16_384 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Tiny_Patch16_384 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Tiny_Patch16_384_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Small_Patch16_224(MBLT_Engine):
    """ViT_Small_Patch16_224 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Small_Patch16_224 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Small_Patch16_224_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Small_Patch16_384(MBLT_Engine):
    """ViT_Small_Patch16_384 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Small_Patch16_384 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Small_Patch16_384_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Small_Patch32_224(MBLT_Engine):
    """ViT_Small_Patch32_224 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Small_Patch32_224 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Small_Patch32_224_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Small_Patch32_384(MBLT_Engine):
    """ViT_Small_Patch32_384 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Small_Patch32_384 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Small_Patch32_384_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Base_Patch8_224(MBLT_Engine):
    """ViT_Base_Patch8_224 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Base_Patch8_224 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Base_Patch8_224_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Base_Patch16_224(MBLT_Engine):
    """ViT_Base_Patch16_224 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Base_Patch16_224 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Base_Patch16_224_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Base_Patch16_384(MBLT_Engine):
    """ViT_Base_Patch16_384 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Base_Patch16_384 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Base_Patch16_384_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Base_Patch32_224(MBLT_Engine):
    """ViT_Base_Patch32_224 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Base_Patch32_224 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Base_Patch32_224_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Base_Patch32_384(MBLT_Engine):
    """ViT_Base_Patch32_384 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Base_Patch32_384 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Base_Patch32_384_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Large_Patch16_224(MBLT_Engine):
    """ViT_Large_Patch16_224 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Large_Patch16_224 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Large_Patch16_224_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Large_Patch16_384(MBLT_Engine):
    """ViT_Large_Patch16_384 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Large_Patch16_384 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Large_Patch16_384_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ViT_Large_Patch32_384(MBLT_Engine):
    """ViT_Large_Patch32_384 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ViT_Large_Patch32_384 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ViT_Large_Patch32_384_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
