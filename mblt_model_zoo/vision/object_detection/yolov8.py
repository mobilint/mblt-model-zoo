from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv8m_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/image_detection/yolov8m.mxq",
                },
                "regulus": {"single": None},
            },
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv8l_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/image_detection/yolov8l.mxq",
                },
                "regulus": {"single": None},
            },
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv8s_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/image_detection/yolov8s.mxq",
                },
                "regulus": {"single": None},
            },
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv8x_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/image_detection/yolov8x.mxq",
                },
                "regulus": {"single": None},
            },
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
            "task": "object_detection",
            "nc": 80,  # Number of classes
            "nl": 3,  # Number of detection layers
        },
    )
    DEFAULT = COCO_V1


class YOLOv8s(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in YOLOv8s_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv8m_Set. Available types: {YOLOv8s_Set.__dict__.keys()}"
        model_cfg = YOLOv8s_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = YOLOv8s_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8s_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8m(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in YOLOv8m_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv8m_Set. Available types: {YOLOv8m_Set.__dict__.keys()}"
        model_cfg = YOLOv8m_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = YOLOv8m_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8m_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8l(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in YOLOv8l_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv8l_Set. Available types: {YOLOv8l_Set.__dict__.keys()}"
        model_cfg = YOLOv8l_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = YOLOv8l_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8l_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)


class YOLOv8x(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in YOLOv8x_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv8x_Set. Available types: {YOLOv8x_Set.__dict__.keys()}"
        model_cfg = YOLOv8x_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = YOLOv8x_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv8x_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
