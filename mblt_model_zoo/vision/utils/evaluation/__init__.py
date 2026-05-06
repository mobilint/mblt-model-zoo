"""
Evaluation scripts for various datasets.
"""

from __future__ import annotations

from .eval_coco import eval_coco
from .eval_imagenet import eval_imagenet

__all__: list[str] = ["eval_coco", "eval_imagenet"]
