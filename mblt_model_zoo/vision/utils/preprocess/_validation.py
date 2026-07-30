"""Validation helpers shared by image preprocessing operations."""

from __future__ import annotations

from collections.abc import Sequence


def normalize_image_size(size: int | Sequence[int], *, name: str = "size") -> list[int]:
    """Normalize a positive scalar or two-dimensional image size to ``[height, width]``."""

    if isinstance(size, bool):
        raise TypeError(f"{name} must be a positive integer or a two-item integer sequence, got bool.")
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"{name} must be positive, got {size}.")
        return [size, size]
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        if len(size) != 2:
            raise ValueError(f"{name} must contain exactly two items, got {size!r}.")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in size):
            raise TypeError(f"{name} items must be integers, got {size!r}.")
        normalized = [int(size[0]), int(size[1])]
        if any(value <= 0 for value in normalized):
            raise ValueError(f"{name} items must be positive, got {size!r}.")
        return normalized
    raise TypeError(f"{name} must be a positive integer or a two-item integer sequence, got {type(size).__name__}.")
