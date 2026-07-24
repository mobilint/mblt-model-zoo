"""Private helpers shared by dense prediction postprocessors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch

from .common import RatioPad


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


def normalize_shapes(img0_shape: tuple[int, int] | Sequence[tuple[int, int]], batch_size: int) -> list[tuple[int, int]]:
    """Normalize one or many original image shapes to a batch-sized list."""

    if len(img0_shape) == 2 and isinstance(img0_shape[0], int):
        return [(int(img0_shape[0]), int(img0_shape[1]))] * batch_size  # type: ignore[index]
    shapes = [(int(shape[0]), int(shape[1])) for shape in img0_shape]  # type: ignore[union-attr]
    if len(shapes) != batch_size:
        raise ValueError(f"Expected {batch_size} original image shapes, got {len(shapes)}.")
    return shapes


def ratio_pad_for_shape(input_shape: tuple[int, int], shape: tuple[int, int]) -> RatioPad:
    """Recreate Ultralytics-compatible letterbox metadata for an original shape."""

    height, width = shape
    ratio = min(input_shape[0] / height, input_shape[1] / width)
    unpadded_width, unpadded_height = int(round(width * ratio)), int(round(height * ratio))
    pad_x = int(round((input_shape[1] - unpadded_width) / 2 - 0.1))
    pad_y = int(round((input_shape[0] - unpadded_height) / 2 - 0.1))
    return ((ratio, ratio), (pad_x, pad_y))


def normalize_ratio_pads(
    ratio_pad: RatioPad | Sequence[RatioPad | None] | None,
    batch_size: int,
    shapes: Sequence[tuple[int, int]],
    input_shape: tuple[int, int],
) -> list[RatioPad]:
    """Normalize explicit or derived letterbox metadata for every sample."""

    if ratio_pad is None:
        return [ratio_pad_for_shape(input_shape, shape) for shape in shapes]
    if isinstance(ratio_pad, tuple) and len(ratio_pad) == 2 and isinstance(ratio_pad[0], tuple):
        return [cast(RatioPad, ratio_pad)] * batch_size
    pads: list[RatioPad] = [
        cast(RatioPad, pad) if pad is not None else ratio_pad_for_shape(input_shape, shapes[index])
        for index, pad in enumerate(ratio_pad)
    ]
    if len(pads) != batch_size:
        raise ValueError(f"Expected {batch_size} ratio_pad values, got {len(pads)}.")
    return pads


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
