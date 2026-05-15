"""Shared numeric helpers for benchmark scripts."""

from __future__ import annotations


def safe_div(numerator: float, denominator: float) -> float | None:
    """Divides two numbers and returns ``None`` when the denominator is zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.

    Returns:
        Division result, or ``None`` for zero denominator.
    """
    if denominator == 0:
        return None
    return numerator / denominator