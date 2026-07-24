"""
ADE20K dataset constants and utilities.
"""

from __future__ import annotations

ADE20K_PALETTE_BGR = [
    (255, 42, 4),
    (235, 219, 11),
    (243, 243, 243),
    (183, 223, 0),
    (104, 31, 17),
    (221, 111, 255),
    (79, 68, 255),
    (0, 237, 204),
    (68, 243, 0),
    (255, 0, 189),
    (255, 180, 0),
    (186, 0, 221),
    (255, 255, 0),
    (0, 192, 38),
    (179, 255, 1),
    (255, 36, 125),
    (104, 0, 123),
    (108, 27, 255),
    (47, 109, 252),
    (11, 255, 162),
]


def get_ade20k_palette(idx: int) -> tuple[int, int, int]:
    """Get an ADE20K visualization color in OpenCV BGR order.

    Args:
        idx: ADE20K class index.

    Returns:
        BGR color tuple. Colors repeat when the index exceeds the palette length.
    """

    return ADE20K_PALETTE_BGR[idx % len(ADE20K_PALETTE_BGR)]
