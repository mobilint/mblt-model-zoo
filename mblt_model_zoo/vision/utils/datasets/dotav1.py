"""
DOTAv1 dataset constants and utilities.
"""

from __future__ import annotations

from ...datasets import get_dataset_class_names

DOTAV1_PALETTE_BGR = [
    (220, 20, 60),
    (119, 11, 32),
    (0, 0, 142),
    (0, 0, 230),
    (106, 0, 228),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 70),
    (0, 0, 192),
    (250, 170, 30),
    (100, 170, 30),
    (220, 220, 0),
    (175, 116, 175),
    (250, 0, 30),
    (165, 42, 42),
]


def get_dotav1_class_num() -> int:
    """Returns the number of DOTAv1 classes.

    Returns:
        Number of DOTAv1 classes.
    """
    return len(get_dataset_class_names("dotav1"))


def get_dotav1_palette(index: int) -> tuple[int, int, int]:
    """Returns the DOTAv1 visualization color for a class index.

    Args:
        index: Class index.

    Returns:
        OpenCV BGR color tuple.
    """

    return DOTAV1_PALETTE_BGR[index]


def get_dotav1_label(index: int) -> str:
    """Returns the DOTAv1 class name for a class index.

    Args:
        index: Class index.

    Returns:
        DOTAv1 class name.
    """
    return get_dataset_class_names("dotav1")[index]
