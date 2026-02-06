"""
YOLOv3 model definitions.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv3_Set(ModelInfoSet):
    """Configuration set for YOLOv3 models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv3",
            "filename": "yolov3.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "anchors": [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326],  # P5/32
            ],
        },
    )
    DEFAULT = COCO_V1


class YOLOv3u_Set(ModelInfoSet):
    """Configuration set for YOLOv3u models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv3u",
            "filename": "yolov3u.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "reg_max": 16,
        },
    )
    DEFAULT = COCO_V1


class YOLOv3_spp_Set(ModelInfoSet):
    """Configuration set for YOLOv3_spp models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv3-spp",
            "filename": "yolov3-spp.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "anchors": [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326],  # P5/32
            ],
        },
    )
    DEFAULT = COCO_V1


class YOLOv3_sppu_Set(ModelInfoSet):
    """Configuration set for YOLOv3_sppu models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv3-sppu",
            "filename": "yolov3-sppu.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "reg_max": 16,
        },
    )
    DEFAULT = COCO_V1


class YOLOv3_tiny_Set(ModelInfoSet):
    """Configuration set for YOLOv3_tiny models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv3-tiny",
            "filename": "yolov3-tiny.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "anchors": [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]],
        },
    )
    DEFAULT = COCO_V1


class YOLOv3_tinyu_Set(ModelInfoSet):
    """Configuration set for YOLOv3_tinyu models."""

    COCO_V1 = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv3-tinyu",
            "filename": "yolov3-tinyu.mxq",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "HWC"},
        },
        post_cfg={
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 2,  # Number of detection layers
            "reg_max": 16,
        },
    )
    DEFAULT = COCO_V1


class YOLOv3(MBLT_Engine):
    """YOLOv3 model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv3 engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv3_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv3u(MBLT_Engine):
    """YOLOv3u model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv3u engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv3u_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv3_spp(MBLT_Engine):
    """YOLOv3_spp model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv3_spp engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv3_spp_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv3_sppu(MBLT_Engine):
    """YOLOv3_sppu model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv3_sppu engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv3_sppu_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv3_tiny(MBLT_Engine):
    """YOLOv3_tiny model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv3_tiny engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv3_tiny_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv3_tinyu(MBLT_Engine):
    """YOLOv3_tinyu model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv3_tinyu engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv3_tinyu_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
