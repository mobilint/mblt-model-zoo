"""Shared core-mode typing and validation helpers."""

from __future__ import annotations

from typing import Literal, cast

CoreMode = Literal["single", "multi", "global4", "global8"]


def normalize_core_mode(core_mode: str) -> CoreMode:
    """Narrow a validated core mode string to the supported literal type.

    Args:
        core_mode: Core mode string from user input or configuration.

    Returns:
        The same value narrowed to ``CoreMode``.

    Raises:
        ValueError: If ``core_mode`` is not one of the supported values.
    """
    valid_modes = {"single", "multi", "global4", "global8"}
    if core_mode not in valid_modes:
        raise ValueError(f"Invalid core mode '{core_mode}'. Expected one of {sorted(valid_modes)}.")
    return cast(CoreMode, core_mode)
