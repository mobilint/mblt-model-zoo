"""Shared file I/O helpers for benchmark scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

JsonObject = dict[str, Any]
CsvRow = dict[str, Any]


def safe_filename(text: str, *, replace_slash_only: bool = False) -> str:
    """Returns a filename-safe representation of text.

    Args:
        text: Source text to convert.
        replace_slash_only: Whether to preserve the legacy behavior of replacing only forward slashes.

    Returns:
        Sanitized text suitable for benchmark output filenames.
    """
    if replace_slash_only:
        return text.replace("/", "__")

    cleaned = text.replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")
    return cleaned or "unnamed"


def write_json(path: Path, payload: Any) -> None:
    """Writes an indented UTF-8 JSON file after creating its parent directory.

    Args:
        path: Destination path.
        payload: JSON-serializable payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Writes dictionaries to CSV using the union of row keys as the field order.

    Args:
        path: Destination path.
        rows: Row mappings to write. Empty rows are ignored to preserve existing script behavior.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})