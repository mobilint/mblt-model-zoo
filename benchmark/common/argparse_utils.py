"""Shared argparse validators for benchmark scripts."""

from __future__ import annotations

import argparse


def parse_positive_int(raw: str) -> int:
    """Parses a positive integer for argparse.

    Args:
        raw: Raw command-line value.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError("expected a positive integer") from e
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def parse_positive_int_optional(raw: str | None) -> int | None:
    """Parses an optional positive integer for argparse.

    Args:
        raw: Raw command-line value or ``None``.

    Returns:
        Parsed positive integer, or ``None`` for empty input.

    Raises:
        argparse.ArgumentTypeError: If the value is not empty and not a positive integer.
    """
    if raw is None or raw == "":
        return None
    return parse_positive_int(raw)


def parse_range_arg(raw: str) -> tuple[int, int, int]:
    """Parses ``start:end:step`` or ``start,end,step`` positive integer ranges.

    Args:
        raw: Raw command-line value.

    Returns:
        Parsed ``(start, end, step)`` tuple.

    Raises:
        argparse.ArgumentTypeError: If the value is malformed.
    """
    sep = ":" if ":" in raw else ("," if "," in raw else None)
    if sep is None:
        raise argparse.ArgumentTypeError("expected format 'start:end:step' or 'start,end,step'")
    parts = [part.strip() for part in raw.split(sep)]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected exactly 3 integers: 'start:end:step' or 'start,end,step'")
    try:
        start, end, step = (int(part) for part in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("range values must be integers") from e
    if start <= 0 or end <= 0 or step <= 0:
        raise argparse.ArgumentTypeError("range values must be positive integers")
    if start > end:
        raise argparse.ArgumentTypeError("range start must be <= end")
    return start, end, step


def parse_int_csv(raw: str, *, unique_sorted: bool = True, allow_empty: bool = False) -> list[int]:
    """Parses comma-separated positive integers.

    Args:
        raw: Raw command-line value.
        unique_sorted: Whether to sort and de-duplicate parsed values.
        allow_empty: Whether an empty input should return an empty list.

    Returns:
        Parsed positive integers.

    Raises:
        argparse.ArgumentTypeError: If the value is malformed.
    """
    parts = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not parts:
        if allow_empty:
            return []
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        values = [int(item) for item in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError("all values must be integers") from e
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return sorted(set(values)) if unique_sorted else values