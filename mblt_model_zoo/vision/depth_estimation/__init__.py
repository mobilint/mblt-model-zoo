"""Depth estimation model exports."""

from __future__ import annotations

from .._compat import create_model_class

__all__: list[str] = ["YOLO26lDepth", "YOLO26mDepth", "YOLO26nDepth", "YOLO26sDepth", "YOLO26xDepth"]

YOLO26lDepth = create_model_class("YOLO26lDepth", __name__)
YOLO26mDepth = create_model_class("YOLO26mDepth", __name__)
YOLO26nDepth = create_model_class("YOLO26nDepth", __name__)
YOLO26sDepth = create_model_class("YOLO26sDepth", __name__)
YOLO26xDepth = create_model_class("YOLO26xDepth", __name__)
