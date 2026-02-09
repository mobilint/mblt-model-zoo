"""
RegNet model definitions.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class RegNet_X_400MF_Set(ModelInfoSet):
    """Configuration set for RegNet X 400MF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_400MF.tv1_in1k",
            "filename": "regnet_x_400mf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_400MF.tv2_in1k",
            "filename": "regnet_x_400mf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_X_800MF_Set(ModelInfoSet):
    """Configuration set for RegNet X 800MF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_800MF.tv1_in1k",
            "filename": "regnet_x_800mf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_800MF.tv2_in1k",
            "filename": "regnet_x_800mf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_X_1_6GF_Set(ModelInfoSet):
    """Configuration set for RegNet X 1.6GF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_1_6GF.tv1_in1k",
            "filename": "regnet_x_1_6gf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_1_6GF.tv2_in1k",
            "filename": "regnet_x_1_6gf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_X_3_2GF_Set(ModelInfoSet):
    """Configuration set for RegNet X 3.2GF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_3_2GF.tv1_in1k",
            "filename": "regnet_x_3_2gf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_3_2GF.tv2_in1k",
            "filename": "regnet_x_3_2gf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_X_8GF_Set(ModelInfoSet):
    """Configuration set for RegNet X 8GF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_8GF.tv1_in1k",
            "filename": "regnet_x_8gf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_8GF.tv2_in1k",
            "filename": "regnet_x_8gf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_X_16GF_Set(ModelInfoSet):
    """Configuration set for RegNet X 16GF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_16GF.tv1_in1k",
            "filename": "regnet_x_16gf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_16GF.tv2_in1k",
            "filename": "regnet_x_16gf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_X_32GF_Set(ModelInfoSet):
    """Configuration set for RegNet X 32GF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_32GF.tv1_in1k",
            "filename": "regnet_x_32gf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_X_32GF.tv2_in1k",
            "filename": "regnet_x_32gf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_Y_400MF_Set(ModelInfoSet):
    """Configuration set for RegNet Y 400MF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_400MF.tv1_in1k",
            "filename": "regnet_y_400mf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_400MF.tv2_in1k",
            "filename": "regnet_y_400mf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_Y_800MF_Set(ModelInfoSet):
    """Configuration set for RegNet Y 800MF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_800MF.tv1_in1k",
            "filename": "regnet_y_800mf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_800MF.tv2_in1k",
            "filename": "regnet_y_800mf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_Y_1_6GF_Set(ModelInfoSet):
    """Configuration set for RegNet Y 1.6GF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_1_6GF.tv1_in1k",
            "filename": "regnet_y_1_6gf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_1_6GF.tv2_in1k",
            "filename": "regnet_y_1_6gf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_Y_3_2GF_Set(ModelInfoSet):
    """Configuration set for RegNet Y 3.2GF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_3_2GF.tv1_in1k",
            "filename": "regnet_y_3_2gf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_3_2GF.tv2_in1k",
            "filename": "regnet_y_3_2gf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_Y_8GF_Set(ModelInfoSet):
    """Configuration set for RegNet Y 8GF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_8GF.tv1_in1k",
            "filename": "regnet_y_8gf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_8GF.tv2_in1k",
            "filename": "regnet_y_8gf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_Y_16GF_Set(ModelInfoSet):
    """Configuration set for RegNet Y 16GF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_16GF.tv1_in1k",
            "filename": "regnet_y_16gf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_16GF.tv2_in1k",
            "filename": "regnet_y_16gf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_Y_32GF_Set(ModelInfoSet):
    """Configuration set for RegNet Y 32GF models."""

    IMAGENET1K_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_32GF.tv1_in1k",
            "filename": "regnet_y_32gf_IMAGENET1K_V1.mxq",
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
    IMAGENET1K_V2 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/RegNet_Y_32GF.tv2_in1k",
            "filename": "regnet_y_32gf_IMAGENET1K_V2.mxq",
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
        post_cfg={"task": "image_classification"},
    )
    DEFAULT = IMAGENET1K_V2  # Default model


class RegNet_X_400MF(MBLT_Engine):
    """RegNet_X_400MF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_X_400MF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_X_400MF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_X_800MF(MBLT_Engine):
    """RegNet_X_800MF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_X_800MF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_X_800MF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_X_1_6GF(MBLT_Engine):
    """RegNet_X_1_6GF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_X_1_6GF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_X_1_6GF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_X_3_2GF(MBLT_Engine):
    """RegNet_X_3_2GF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_X_3_2GF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_X_3_2GF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_X_8GF(MBLT_Engine):
    """RegNet_X_8GF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_X_8GF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_X_8GF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_X_16GF(MBLT_Engine):
    """RegNet_X_16GF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_X_16GF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_X_16GF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_X_32GF(MBLT_Engine):
    """RegNet_X_32GF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_X_32GF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_X_32GF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_Y_400MF(MBLT_Engine):
    """RegNet_Y_400MF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_Y_400MF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_Y_400MF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_Y_800MF(MBLT_Engine):
    """RegNet_Y_800MF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_Y_800MF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_Y_800MF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_Y_1_6GF(MBLT_Engine):
    """RegNet_Y_1_6GF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_Y_1_6GF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_Y_1_6GF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_Y_3_2GF(MBLT_Engine):
    """RegNet_Y_3_2GF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_Y_3_2GF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_Y_3_2GF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_Y_8GF(MBLT_Engine):
    """RegNet_Y_8GF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_Y_8GF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_Y_8GF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_Y_16GF(MBLT_Engine):
    """RegNet_Y_16GF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_Y_16GF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_Y_16GF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class RegNet_Y_32GF(MBLT_Engine):
    """RegNet_Y_32GF model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the RegNet_Y_32GF engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            RegNet_Y_32GF_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
