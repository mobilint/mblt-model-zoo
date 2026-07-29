"""Private helpers shared by dense prediction postprocessors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..letterbox import LetterBoxGeometry, RatioPad, resolve_ratio_pad
from .common import normalize_ratio_pads


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
    return [resolve_ratio_pad(input_shape, shape, pad) for pad, shape in zip(pads, shapes)]


def crop_letterbox(
    output: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: RatioPad,
    input_shape: tuple[int, int],
    task_name: str,
) -> torch.Tensor:
    """Crop letterbox padding from a dense two-dimensional output."""

    geometry = LetterBoxGeometry.from_shapes(input_shape, shape)
    output_shape = (int(output.shape[0]), int(output.shape[1]))
    top, bottom, left, right = geometry.crop_bounds(output_shape, pad=ratio_pad[1])
    cropped = output[top:bottom, left:right]
    if cropped.numel() == 0:
        raise ValueError(f"{task_name} letterbox restoration produced an empty crop.")
    return cropped
