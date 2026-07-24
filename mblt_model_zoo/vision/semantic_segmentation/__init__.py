"""Semantic segmentation model exports."""

from __future__ import annotations

from .._compat import create_model_class

__all__: list[str] = [
    "YOLO26lSem",
    "YOLO26lSemADE20K",
    "YOLO26mSem",
    "YOLO26mSemADE20K",
    "YOLO26nSem",
    "YOLO26nSemADE20K",
    "YOLO26sSem",
    "YOLO26sSemADE20K",
    "YOLO26xSem",
    "YOLO26xSemADE20K",
]

YOLO26lSem = create_model_class("YOLO26lSem", __name__)
YOLO26lSemADE20K = create_model_class("YOLO26lSemADE20K", __name__)
YOLO26mSem = create_model_class("YOLO26mSem", __name__)
YOLO26mSemADE20K = create_model_class("YOLO26mSemADE20K", __name__)
YOLO26nSem = create_model_class("YOLO26nSem", __name__)
YOLO26nSemADE20K = create_model_class("YOLO26nSemADE20K", __name__)
YOLO26sSem = create_model_class("YOLO26sSem", __name__)
YOLO26sSemADE20K = create_model_class("YOLO26sSemADE20K", __name__)
YOLO26xSem = create_model_class("YOLO26xSem", __name__)
YOLO26xSemADE20K = create_model_class("YOLO26xSemADE20K", __name__)
