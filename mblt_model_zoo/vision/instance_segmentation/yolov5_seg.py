from ..utils.types import ModelInfo, ModelInfoSet
from ..wrapper import MBLT_Engine


class YOLOv5lSeg_Set(ModelInfoSet):
    COCO_V1 = ModelInfo(
        model_cfg={
            "url_dict": {
                "aries": {
                    "single": None,
                    "multi": None,
                    "global": "https://dl.mobilint.com/model/image_detection/yolov5l-seg.mxq",
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
            "task": "instance_segmentation",
            "nc": 80,  # Number of classes
            "anchors": [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326],  # P5/32
            ],
            "n_extra": 32,
        },
    )
    DEFAULT = COCO_V1


class YOLOv5lSeg(MBLT_Engine):
    def __init__(
        self,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        assert (
            model_type in YOLOv5lSeg_Set.__dict__.keys()
        ), f"Model type {model_type} not found in YOLOv5lSeg_Set. Available types: {YOLOv5lSeg_Set.__dict__.keys()}"
        model_cfg = YOLOv5lSeg_Set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product
        pre_cfg = YOLOv5lSeg_Set.__dict__[model_type].value.pre_cfg
        post_cfg = YOLOv5lSeg_Set.__dict__[model_type].value.post_cfg
        super().__init__(model_cfg, pre_cfg, post_cfg)
