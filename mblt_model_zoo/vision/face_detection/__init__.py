"""Face detection exports for the vision package."""

from __future__ import annotations

from .._compat import create_model_class

__all__: list[str] = [
    "YOLO11l_face",
    "YOLO11m_face",
    "YOLO11n_face",
    "YOLO11s_face",
    "YOLO12l_face",
    "YOLO12m_face",
    "YOLO12n_face",
    "YOLO12s_face",
    "YOLOv10l_face",
    "YOLOv10m_face",
    "YOLOv10n_face",
    "YOLOv10s_face",
    "YOLOv6m_face",
    "YOLOv6n_face",
    "YOLOv8l_face",
    "YOLOv8m_face",
    "YOLOv8n_face",
]

YOLO11l_face = create_model_class("YOLO11l_face", __name__)
YOLO11m_face = create_model_class("YOLO11m_face", __name__)
YOLO11n_face = create_model_class("YOLO11n_face", __name__)
YOLO11s_face = create_model_class("YOLO11s_face", __name__)
YOLO12l_face = create_model_class("YOLO12l_face", __name__)
YOLO12m_face = create_model_class("YOLO12m_face", __name__)
YOLO12n_face = create_model_class("YOLO12n_face", __name__)
YOLO12s_face = create_model_class("YOLO12s_face", __name__)
YOLOv10l_face = create_model_class("YOLOv10l_face", __name__)
YOLOv10m_face = create_model_class("YOLOv10m_face", __name__)
YOLOv10n_face = create_model_class("YOLOv10n_face", __name__)
YOLOv10s_face = create_model_class("YOLOv10s_face", __name__)
YOLOv6m_face = create_model_class("YOLOv6m_face", __name__)
YOLOv6n_face = create_model_class("YOLOv6n_face", __name__)
YOLOv8l_face = create_model_class("YOLOv8l_face", __name__)
YOLOv8m_face = create_model_class("YOLOv8m_face", __name__)
YOLOv8n_face = create_model_class("YOLOv8n_face", __name__)
