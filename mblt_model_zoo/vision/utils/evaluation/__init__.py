"""
Evaluation scripts for various datasets.
"""

from .eval_coco import eval_coco
from .eval_imagenet import eval_imagenet

__all__ = ["eval_coco", "eval_imagenet"]
