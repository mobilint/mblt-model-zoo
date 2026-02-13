"""
YOLOv8 Pose Estimation model definitions.
"""

from typing import Optional

from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv8nPose_Set(ModelInfoSet):
    """Configuration set for YOLOv8nPose models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv8n-pose",
            "filename": "yolov8n-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
            "reg_max": 16,
        },
    )


class YOLOv8sPose_Set(ModelInfoSet):
    """Configuration set for YOLOv8sPose models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv8s-pose",
            "filename": "yolov8s-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
            "reg_max": 16,
        },
    )


class YOLOv8mPose_Set(ModelInfoSet):
    """Configuration set for YOLOv8mPose models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv8m-pose",
            "filename": "yolov8m-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
            "reg_max": 16,
        },
    )


class YOLOv8lPose_Set(ModelInfoSet):
    """Configuration set for YOLOv8lPose models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv8l-pose",
            "filename": "yolov8l-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
            "reg_max": 16,
        },
    )


class YOLOv8xPose_Set(ModelInfoSet):
    """Configuration set for YOLOv8xPose models."""

    DEFAULT = ModelInfo(
        model_cfg={
            "repo_id": "mobilint/YOLOv8x-pose",
            "filename": "yolov8x-pose.mxq",
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
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
            "reg_max": 16,
        },
    )


class YOLOv8nPose(MBLT_Engine):
    """YOLOv8nPose model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv8nPose engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv8nPose_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8sPose(MBLT_Engine):
    """YOLOv8sPose model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv8sPose engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv8sPose_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8mPose(MBLT_Engine):
    """YOLOv8mPose model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv8mPose engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv8mPose_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8lPose(MBLT_Engine):
    """YOLOv8lPose model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv8lPose engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv8lPose_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8xPose(MBLT_Engine):
    """YOLOv8xPose model engine."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Initializes the YOLOv8xPose engine.

        Args:
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".
        """
        model_cfg, pre_cfg, post_cfg = self._get_configs(
            YOLOv8xPose_Set,
            local_path=local_path,
            model_type=model_type,
            infer_mode=infer_mode,
            product=product,
        )
        super().__init__(model_cfg, pre_cfg, post_cfg)
