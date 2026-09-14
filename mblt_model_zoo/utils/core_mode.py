"""Shared core-mode typing and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

CoreMode = Literal["auto", "single", "multi", "global4", "global8"]
CORE_MODE_VALUES = frozenset({"auto", "single", "multi", "global4", "global8"})
BATCH_CORE_MODE_VALUES = frozenset({"auto", "single"})


def normalize_core_mode(core_mode: str) -> CoreMode:
    """Narrow a validated core mode string to the supported literal type.

    Args:
        core_mode: Core mode string from user input or configuration.

    Returns:
        The same value narrowed to ``CoreMode``.

    Raises:
        ValueError: If ``core_mode`` is not one of the supported values.
    """
    if core_mode not in CORE_MODE_VALUES:
        raise ValueError(f"Invalid core mode '{core_mode}'. Expected one of {sorted(CORE_MODE_VALUES)}.")
    return cast(CoreMode, core_mode)


def validate_batch_core_mode(core_mode: CoreMode) -> CoreMode:
    """Validate a mode supported by batched MXQ execution."""
    if core_mode not in BATCH_CORE_MODE_VALUES:
        raise ValueError(f"Batch execution only supports core mode single or auto, got '{core_mode}'.")
    return core_mode


def normalize_config_core_mode(value: Any) -> CoreMode | None:
    """Normalize an optional raw config mode without treating invalid values as explicit."""
    if not isinstance(value, str):
        return None
    value = value.strip().casefold()
    return cast(CoreMode, value) if value in CORE_MODE_VALUES else None


def config_core_mode_candidates(payload: Mapping[str, Any], *, role: str = "shared") -> list[Any]:
    """Return raw config mode candidates in the canonical role-aware order.

    ``text`` and ``base`` are LLM roles. Their nested mode wins over the legacy
    top-level field; ``vision`` uses its nested field before that same fallback.
    """
    candidates: list[Any] = []
    if role == "base":
        candidates.append(payload.get("base_core_mode"))
    elif role == "text":
        text_config = payload.get("text_config")
        if isinstance(text_config, Mapping):
            candidates.append(text_config.get("core_mode"))
    elif role == "vision":
        vision_config = payload.get("vision_config")
        if isinstance(vision_config, Mapping):
            candidates.append(vision_config.get("core_mode"))
    candidates.append(payload.get("core_mode"))
    return candidates


def resolve_config_core_mode(
    payload: Mapping[str, Any] | None,
    *,
    role: str = "shared",
    fallback: CoreMode = "auto",
) -> CoreMode:
    """Resolve a config mode with canonical role precedence and an ``auto`` fallback."""
    if payload is not None:
        for candidate in config_core_mode_candidates(payload, role=role):
            mode = normalize_config_core_mode(candidate)
            if mode is not None:
                return mode
    return fallback
