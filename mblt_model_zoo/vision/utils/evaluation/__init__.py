"""
Evaluation scripts for various datasets.
"""

from __future__ import annotations

from .eval_coco import eval_coco
from .eval_dota import eval_dota
from .eval_imagenet import eval_imagenet
from .eval_widerface import WiderFaceResult, eval_widerface

__all__: list[str] = ["eval_coco", "eval_dota", "eval_imagenet", "WiderFaceResult", "eval_widerface"]
