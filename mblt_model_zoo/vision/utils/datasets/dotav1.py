"""
DOTAv1 dataset constants and utilities.
"""

from __future__ import annotations

CLASSES = (
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
)


def get_dotav1_class_num() -> int:
    """Returns the number of DOTAv1 classes.

    Returns:
        Number of DOTAv1 classes.
    """
    return len(CLASSES)


def get_dotav1_label(index: int) -> str:
    """Returns the DOTAv1 class name for a class index.

    Args:
        index: Class index.

    Returns:
        DOTAv1 class name.
    """
    return CLASSES[index]
