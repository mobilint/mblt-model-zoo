from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv8nPose_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8sPose_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8mPose_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8lPose_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8xPose_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "/",
        },
        pre_cfg={
            "Reader": {
                "style": "numpy",
            },
            "YoloPre": {
                "img_size": [640, 640],
            },
            "SetOrder": {"shape": "CHW"},
        },
        post_cfg={
            "task": "pose_estimation",
            "nc": 1,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 51,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8nPose(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8nPose_Set.__dict__.keys()
        ), f"model_type {model_type} not found in YOLOv8nPose_Set. Available types: {YOLOv8nPose_Set.__dict__.keys()}"
        model_cfg = YOLOv8nPose_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8nPose_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8nPose_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8sPose(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8sPose_Set.__dict__.keys()
        ), f"model_type {model_type} not found in YOLOv8sPose_Set. Available types: {YOLOv8sPose_Set.__dict__.keys()}"
        model_cfg = YOLOv8sPose_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8sPose_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8sPose_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8mPose(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8mPose_Set.__dict__.keys()
        ), f"model_type {model_type} not found in YOLOv8mPose_Set. Available types: {YOLOv8mPose_Set.__dict__.keys()}"
        model_cfg = YOLOv8mPose_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8mPose_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8mPose_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8lPose(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8lPose_Set.__dict__.keys()
        ), f"model_type {model_type} not found in YOLOv8lPose_Set. Available types: {YOLOv8lPose_Set.__dict__.keys()}"
        model_cfg = YOLOv8lPose_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8lPose_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8lPose_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8xPose(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8xPose_Set.__dict__.keys()
        ), f"model_type {model_type} not found in YOLOv8xPose_Set. Available types: {YOLOv8xPose_Set.__dict__.keys()}"
        model_cfg = YOLOv8xPose_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8xPose_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8xPose_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
