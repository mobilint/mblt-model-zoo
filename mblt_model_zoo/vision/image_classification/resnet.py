"""
ResNet model definitions (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152).
"""

from collections import OrderedDict
from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class ResNet18_Set(ModelInfoSet):
    """Configuration set for ResNet18 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/ResNet18",
                "filename": "resnet18_IMAGENET1K_V1.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
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
                "Normalize": {
                    "style": "torch",
                },
            }
        ),
        post_cfg=OrderedDict(
            {
                "task": "image_classification",
            }
        ),
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class ResNet34_Set(ModelInfoSet):
    """Configuration set for ResNet34 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/ResNet34",
                "filename": "resnet34_IMAGENET1K_V1.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
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
                "Normalize": {
                    "style": "torch",
                },
            }
        ),
        post_cfg=OrderedDict(
            {
                "task": "image_classification",
            }
        ),
    )
    DEFAULT = IMAGENET1K_V1  # Default model


class ResNet50_Set(ModelInfoSet):
    """Configuration set for ResNet50 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/ResNet50.tv1_in1k",
                "filename": "resnet50_IMAGENET1K_V1.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
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
                "Normalize": {
                    "style": "torch",
                },
            }
        ),
        post_cfg=OrderedDict(
            {
                "task": "image_classification",
            }
        ),
    )
    IMAGENET1K_V2 = IMAGENET1K_V1.update_model_cfg(
        repo_id="mobilint/ResNet50.tv2_in1k",
        filename="resnet50_IMAGENET1K_V2.mxq",
    ).update_pre_cfg(
        Resize={
            "size": 232,
            "interpolation": "bilinear",
        },
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class ResNet101_Set(ModelInfoSet):
    """Configuration set for ResNet101 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/ResNet101.tv1_in1k",
                "filename": "resnet101_IMAGENET1K_V1.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
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
                "Normalize": {
                    "style": "torch",
                },
            }
        ),
        post_cfg=OrderedDict(
            {
                "task": "image_classification",
            }
        ),
    )
    IMAGENET1K_V2 = IMAGENET1K_V1.update_model_cfg(
        repo_id="mobilint/ResNet101.tv2_in1k",
        filename="resnet101_IMAGENET1K_V2.mxq",
    ).update_pre_cfg(
        Resize={
            "size": 232,
            "interpolation": "bilinear",
        },
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class ResNet152_Set(ModelInfoSet):
    """Configuration set for ResNet152 models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg=OrderedDict(
            {
                "repo_id": "mobilint/ResNet152.tv1_in1k",
                "filename": "resnet152_IMAGENET1K_V1.mxq",
                "revision": "main",
            }
        ),
        pre_cfg=OrderedDict(
            {
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
                "Normalize": {
                    "style": "torch",
                },
            }
        ),
        post_cfg=OrderedDict(
            {
                "task": "image_classification",
            }
        ),
    )
    IMAGENET1K_V2 = IMAGENET1K_V1.update_model_cfg(
        repo_id="mobilint/ResNet152.tv2_in1k",
        filename="resnet152_IMAGENET1K_V2.mxq",
    ).update_pre_cfg(
        Resize={
            "size": 232,
            "interpolation": "bilinear",
        },
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class ResNet18(MBLT_Engine):
    """ResNet18 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ResNet18 engine.

        Args:
                local_path: Path to a local model file. Defaults to None.
                model_type: Model configuration type. Defaults to "DEFAULT".
                infer_mode: Inference execution mode. Defaults to "global8".
                product: Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ResNet18_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ResNet34(MBLT_Engine):
    """ResNet34 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ResNet34 engine.

        Args:
                local_path: Path to a local model file. Defaults to None.
                model_type: Model configuration type. Defaults to "DEFAULT".
                infer_mode: Inference execution mode. Defaults to "global8".
                product: Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ResNet34_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ResNet50(MBLT_Engine):
    """ResNet50 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ResNet50 engine.

        Args:
                local_path: Path to a local model file. Defaults to None.
                model_type: Model configuration type. Defaults to "DEFAULT".
                infer_mode: Inference execution mode. Defaults to "global8".
                product: Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ResNet50_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ResNet101(MBLT_Engine):
    """ResNet101 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ResNet101 engine.

        Args:
                local_path: Path to a local model file. Defaults to None.
                model_type: Model configuration type. Defaults to "DEFAULT".
                infer_mode: Inference execution mode. Defaults to "global8".
                product: Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ResNet101_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class ResNet152(MBLT_Engine):
    """ResNet152 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the ResNet152 engine.

        Args:
                local_path: Path to a local model file. Defaults to None.
                model_type: Model configuration type. Defaults to "DEFAULT".
                infer_mode: Inference execution mode. Defaults to "global8".
                product: Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            ResNet152_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
