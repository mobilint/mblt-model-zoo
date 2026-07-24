"""Private helpers shared by dense prediction postprocessors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .common import RatioPad, compute_ratio_pad, normalize_ratio_pads


def get_letterbox_input_shape(
    pre_cfg: dict[str, Any],
    requirement_name: str,
    size_name: str | None = None,
) -> tuple[int, int]:
    """Validate and return a dense task's configured letterbox input shape."""

    letterbox_cfg = pre_cfg.get("LetterBox")
    if not isinstance(letterbox_cfg, dict) or "img_size" not in letterbox_cfg:
        raise ValueError(f"{requirement_name} requires a LetterBox configuration in pre_cfg.")
    image_size = letterbox_cfg["img_size"]
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError(f"{size_name or requirement_name} LetterBox img_size must be a two-item [height, width] list.")
    return int(image_size[0]), int(image_size[1])


def resolve_ratio_pads(
    ratio_pad: RatioPad | Sequence[RatioPad | None] | None,
    batch_size: int,
    shapes: Sequence[tuple[int, int]],
    input_shape: tuple[int, int],
) -> list[RatioPad]:
    """Normalize letterbox metadata and derive values missing from a dense task batch."""

    pads = normalize_ratio_pads(ratio_pad, batch_size)
    resolved = []
    for pad, shape in zip(pads, shapes):
        if pad is not None:
            resolved.append(pad)
            continue
        gain, offset = compute_ratio_pad(input_shape, shape)
        resolved.append(((gain, gain), offset))
    return resolved


def crop_letterbox(
    output: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: RatioPad,
    input_shape: tuple[int, int],
    task_name: str,
) -> torch.Tensor:
    """Crop letterbox padding from a dense two-dimensional output."""

    ratio, pad = ratio_pad
    del ratio
    output_height, output_width = output.shape
    scale_x = output_width / input_shape[1]
    scale_y = output_height / input_shape[0]
    left = int(round(pad[0] * scale_x))
    top = int(round(pad[1] * scale_y))
    original_height, original_width = shape
    resize_ratio = min(input_shape[0] / original_height, input_shape[1] / original_width)
    unpadded_width = int(round(original_width * resize_ratio * scale_x))
    unpadded_height = int(round(original_height * resize_ratio * scale_y))
    cropped = output[top : top + unpadded_height, left : left + unpadded_width]
    if cropped.numel() == 0:
        raise ValueError(f"{task_name} letterbox restoration produced an empty crop.")
    return cropped
