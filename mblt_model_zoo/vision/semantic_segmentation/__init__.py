"""Semantic segmentation model exports."""

from __future__ import annotations

from .._compat import create_model_class

__all__: list[str] = [
    "YOLO26lSemADE20K",
    "YOLO26mSemADE20K",
    "YOLO26nSemADE20K",
    "YOLO26sSemADE20K",
    "YOLO26xSemADE20K",
]

YOLO26lSemADE20K = create_model_class("YOLO26lSemADE20K", __name__)
YOLO26mSemADE20K = create_model_class("YOLO26mSemADE20K", __name__)
YOLO26nSemADE20K = create_model_class("YOLO26nSemADE20K", __name__)
YOLO26sSemADE20K = create_model_class("YOLO26sSemADE20K", __name__)
YOLO26xSemADE20K = create_model_class("YOLO26xSemADE20K", __name__)
