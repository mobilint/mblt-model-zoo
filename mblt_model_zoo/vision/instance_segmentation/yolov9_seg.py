"""
YOLOv9 Segmentation model definitions.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class GELANcSeg_Set(ModelInfoSet):
    """Configuration set for GELANcSeg models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/GELANc-seg",
            "filename": "gelanc-seg.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )


class YOLOv9cSeg_Set(ModelInfoSet):
    """Configuration set for YOLOv9cSeg models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv9c-seg",
            "filename": "yolov9c-seg.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )


class YOLOv9eSeg_Set(ModelInfoSet):
    """Configuration set for YOLOv9eSeg models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv9e-seg",
            "filename": "yolov9e-seg.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "LetterBox": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
            "reg_max": 16,
        },
    )


class GELANcSeg(MBLT_Engine):
    """GELANcSeg model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the GELANcSeg engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            GELANcSeg_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv9cSeg(MBLT_Engine):
    """YOLOv9cSeg model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv9cSeg engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv9cSeg_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv9eSeg(MBLT_Engine):
    """YOLOv9eSeg model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv9eSeg engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv9eSeg_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
