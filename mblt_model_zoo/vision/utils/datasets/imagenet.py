"""
ImageNet dataset constants and utilities.
"""

from __future__ import annotations

from ...datasets import get_dataset_class_names


def get_imagenet_label(idx: int) -> str:
    """Get the descriptive label for an ImageNet class index.

    Args:
        idx (int): ImageNet class index (0-999).

    Returns:
        str: Primary class label string.
    """
    label_str = get_dataset_class_names("imagenet")[idx]
    return label_str.split(",")[0]


def get_imagenet_class_num() -> int:
    """Get the total number of classes in the ImageNet dataset.

    Returns:
        int: Total number of ImageNet classes.
    """
    return len(get_dataset_class_names("imagenet"))
