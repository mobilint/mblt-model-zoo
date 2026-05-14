"""Oriented Bounding Box (OBB) model exports."""

from __future__ import annotations

from .._compat import create_model_class

__all__: list[str] = [
    "YOLO11lObb",
    "YOLO11mObb",
    "YOLO11nObb",
    "YOLO11sObb",
    "YOLO11xObb",
    "YOLO26lObb",
    "YOLO26mObb",
    "YOLO26nObb",
    "YOLO26sObb",
    "YOLO26xObb",
    "YOLOv8lObb",
    "YOLOv8mObb",
    "YOLOv8nObb",
    "YOLOv8sObb",
    "YOLOv8xObb",
]

YOLO11lObb = create_model_class("YOLO11lObb", __name__)
YOLO11mObb = create_model_class("YOLO11mObb", __name__)
YOLO11nObb = create_model_class("YOLO11nObb", __name__)
YOLO11sObb = create_model_class("YOLO11sObb", __name__)
YOLO11xObb = create_model_class("YOLO11xObb", __name__)
YOLO26lObb = create_model_class("YOLO26lObb", __name__)
YOLO26mObb = create_model_class("YOLO26mObb", __name__)
YOLO26nObb = create_model_class("YOLO26nObb", __name__)
YOLO26sObb = create_model_class("YOLO26sObb", __name__)
YOLO26xObb = create_model_class("YOLO26xObb", __name__)
YOLOv8lObb = create_model_class("YOLOv8lObb", __name__)
YOLOv8mObb = create_model_class("YOLOv8mObb", __name__)
YOLOv8nObb = create_model_class("YOLOv8nObb", __name__)
YOLOv8sObb = create_model_class("YOLOv8sObb", __name__)
YOLOv8xObb = create_model_class("YOLOv8xObb", __name__)
