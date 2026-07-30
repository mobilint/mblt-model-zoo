"""Cityscapes dataset visualization utilities."""

from __future__ import annotations

import numpy as np

# OpenCV images use BGR channel order. These are the official Cityscapes train-ID
# colors converted from RGB so familiar classes retain their standard appearance.
CITYSCAPES_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (128, 64, 128),  # road
    (232, 35, 244),  # sidewalk
    (70, 70, 70),  # building
    (156, 102, 102),  # wall
    (153, 153, 190),  # fence
    (153, 153, 153),  # pole
    (30, 170, 250),  # traffic light
    (0, 220, 220),  # traffic sign
    (35, 142, 107),  # vegetation
    (152, 251, 152),  # terrain
    (180, 130, 70),  # sky
    (60, 20, 220),  # person
    (0, 0, 255),  # rider
    (142, 0, 0),  # car
    (70, 0, 0),  # truck
    (100, 60, 0),  # bus
    (100, 80, 0),  # train
    (230, 0, 0),  # motorcycle
    (32, 11, 119),  # bicycle
)

CITYSCAPES_SOURCE_IDS: tuple[int, ...] = (7, 8, 11, 12, 13, 17, *range(19, 29), 31, 32, 33)
CITYSCAPES_SOURCE_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
CITYSCAPES_SOURCE_TO_TRAIN_ID[list(CITYSCAPES_SOURCE_IDS)] = np.arange(19, dtype=np.uint8)


def get_cityscapes_palette(class_id: int) -> tuple[int, int, int]:
    """Return the BGR visualization color for a Cityscapes train ID.

    Args:
        class_id: Cityscapes train ID in the range 0 through 18.

    Returns:
        The corresponding color in OpenCV BGR order.

    Raises:
        ValueError: If ``class_id`` is outside the Cityscapes train-ID range.
    """

    if not 0 <= class_id < len(CITYSCAPES_PALETTE_BGR):
        raise ValueError(f"Cityscapes class ID must be in [0, {len(CITYSCAPES_PALETTE_BGR) - 1}].")
    return CITYSCAPES_PALETTE_BGR[class_id]
