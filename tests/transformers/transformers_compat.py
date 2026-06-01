"""Helpers for version-specific Transformers test compatibility guards."""

from __future__ import annotations

from importlib import metadata

import pytest


def _parse_version(version: str) -> tuple[int, ...]:
    """Return a numeric tuple from a dotted version string."""
    parts: list[int] = []
    for chunk in version.split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def installed_transformers_version() -> tuple[int, ...]:
    """Return the installed Transformers version as a numeric tuple."""
    return _parse_version(metadata.version("transformers"))


def is_transformers_version_between(min_version: str, max_version: str) -> bool:
    """Return whether the installed Transformers version is within an inclusive range."""
    current = installed_transformers_version()
    return _parse_version(min_version) <= current <= _parse_version(max_version)


def is_transformers_version(version: str) -> bool:
    """Return whether the installed Transformers version matches exactly."""
    return installed_transformers_version() == _parse_version(version)


def skip_if_transformers_version_between(min_version: str, max_version: str, reason: str) -> None:
    """Skip the current module when the installed Transformers version is within a range."""
    if is_transformers_version_between(min_version, max_version):
        pytest.skip(reason, allow_module_level=True)


def skip_if_transformers_version(version: str, reason: str) -> None:
    """Skip the current module when the installed Transformers version matches exactly."""
    if is_transformers_version(version):
        pytest.skip(reason, allow_module_level=True)