"""Pose estimation model exports."""

from .._compat import create_model_class

__all__ = [
    "YOLO11lPose",
    "YOLO11mPose",
    "YOLO11nPose",
    "YOLO11sPose",
    "YOLO11xPose",
    "YOLO26lPose",
    "YOLO26mPose",
    "YOLO26nPose",
    "YOLO26sPose",
    "YOLO26xPose",
    "YOLOv8lPose",
    "YOLOv8mPose",
    "YOLOv8nPose",
    "YOLOv8sPose",
    "YOLOv8xPose",
    "YOLOv8xPoseP6",
]

YOLO11lPose = create_model_class("YOLO11lPose", __name__)
YOLO11mPose = create_model_class("YOLO11mPose", __name__)
YOLO11nPose = create_model_class("YOLO11nPose", __name__)
YOLO11sPose = create_model_class("YOLO11sPose", __name__)
YOLO11xPose = create_model_class("YOLO11xPose", __name__)
YOLO26lPose = create_model_class("YOLO26lPose", __name__)
YOLO26mPose = create_model_class("YOLO26mPose", __name__)
YOLO26nPose = create_model_class("YOLO26nPose", __name__)
YOLO26sPose = create_model_class("YOLO26sPose", __name__)
YOLO26xPose = create_model_class("YOLO26xPose", __name__)
YOLOv8lPose = create_model_class("YOLOv8lPose", __name__)
YOLOv8mPose = create_model_class("YOLOv8mPose", __name__)
YOLOv8nPose = create_model_class("YOLOv8nPose", __name__)
YOLOv8sPose = create_model_class("YOLOv8sPose", __name__)
YOLOv8xPose = create_model_class("YOLOv8xPose", __name__)
YOLOv8xPoseP6 = create_model_class("YOLOv8xPoseP6", __name__)
