"""
DOTAv1 dataset constants and utilities.
"""

from __future__ import annotations

from ...datasets import get_dataset_class_names


def get_dotav1_class_num() -> int:
    """Returns the number of DOTAv1 classes.

    Returns:
        Number of DOTAv1 classes.
    """
    return len(get_dataset_class_names("dotav1"))


def get_dotav1_label(index: int) -> str:
    """Returns the DOTAv1 class name for a class index.

    Args:
        index: Class index.

    Returns:
        DOTAv1 class name.
    """
    return get_dataset_class_names("dotav1")[index]
