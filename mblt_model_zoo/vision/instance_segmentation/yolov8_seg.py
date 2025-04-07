from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv8sSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "dl.mobilint.com/model/image_detection/yolov8s-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8mSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "dl.mobilint.com/model/image_detection/yolov8m-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8lSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url": "dl.mobilint.com/model/image_detection/yolov8l-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


class YOLOv8sSeg(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8sSeg_Set.__dict__.keys()
        ), f"model_type {model_type} not found in YOLOv8sSeg_Set. Available types: {YOLOv8sSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv8sSeg_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8sSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8sSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8mSeg(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8mSeg_Set.__dict__.keys()
        ), f"model_type {model_type} not found in YOLOv8mSeg_Set. Available types: {YOLOv8mSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv8mSeg_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8mSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8mSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8lSeg(MBLT_Engine):
    def __init__(self, local_model: str = None, model_type: str = "DEFAULT"):
        assert (
            model_type in YOLOv8lSeg_Set.__dict__.keys()
        ), f"model_type {model_type} not found in YOLOv8lSeg_Set. Available types: {YOLOv8lSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv8lSeg_Set.__dict__[model_type].value.model_cfg
        if local_model is not None:
            model_cfg["url"] = local_model
        pre_cfg = YOLOv8lSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8lSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
