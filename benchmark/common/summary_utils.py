"""Markdown summary helpers for benchmark outputs."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


HOST_PC_INFO_FILENAME = "host_pc_info.json"


def collect_host_pc_info(results_dir: Path | str, *, filename: str = HOST_PC_INFO_FILENAME) -> Path:
    """Run ``mblt-tracker collect`` and save its JSON output.

    The benchmark should not fail just because host information collection is unavailable. Failures are therefore
    converted into a JSON payload that can still be rendered in the summary.

    Args:
        results_dir: Benchmark output directory.
        filename: JSON filename to write under ``results_dir``.

    Returns:
        Path to the written JSON file.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename
    payload: Any
    try:
        proc = subprocess.run(
            ["mblt-tracker", "collect"],
            check=False,
            capture_output=True,
            encoding="utf-8",
            timeout=30,
        )
        if proc.returncode == 0:
            try:
                payload = json.loads(proc.stdout)
            except json.JSONDecodeError:
                payload = {
                    "status": "error",
                    "message": "mblt-tracker collect did not return valid JSON.",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode,
                }
        else:
            payload = {
                "status": "error",
                "message": "mblt-tracker collect failed.",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
    except FileNotFoundError:
        payload = {"status": "error", "message": "mblt-tracker CLI was not found."}
    except subprocess.TimeoutExpired as e:
        payload = {
            "status": "error",
            "message": "mblt-tracker collect timed out.",
            "stdout": e.stdout,
            "stderr": e.stderr,
            "timeout_s": e.timeout,
        }

    payload = _with_collection_metadata(payload)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved Host PC Info: {output_path.name}")
    return output_path


def write_summary_markdown(
    path: Path | str,
    *,
    title: str,
    host_info_path: Path | str | None,
    table_markdown_path: Path | str | None,
    plot_paths: Sequence[Path | str],
) -> None:
    """Write a benchmark summary Markdown file with host info, plots, and table.

    Args:
        path: Summary Markdown destination.
        title: Document title.
        host_info_path: JSON file written by :func:`collect_host_pc_info`.
        table_markdown_path: Existing Markdown table to include.
        plot_paths: PNG files to embed in the summary.
    """
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}\n\n"]
    lines.extend(_host_info_markdown(Path(host_info_path) if host_info_path else None))
    lines.extend(_plots_markdown(summary_path.parent, [Path(p) for p in plot_paths]))
    lines.extend(_table_markdown(Path(table_markdown_path) if table_markdown_path else None))
    summary_path.write_text("".join(lines), encoding="utf-8")


def existing_png_paths(results_dir: Path | str, *, prefixes: Sequence[str] | None = None) -> list[Path]:
    """Return sorted PNG paths under a benchmark results directory.

    Args:
        results_dir: Directory to scan.
        prefixes: Optional filename prefixes to include.

    Returns:
        Sorted PNG files matching the optional prefixes.
    """
    out_dir = Path(results_dir)
    paths = sorted(out_dir.glob("*.png"))
    if prefixes is None:
        return paths
    prefix_tuple = tuple(prefixes)
    return [path for path in paths if path.name.startswith(prefix_tuple)]


def _with_collection_metadata(payload: Any) -> dict[str, Any]:
    collected_at = datetime.now().astimezone().isoformat(timespec="seconds")
    if isinstance(payload, dict):
        out = dict(payload)
        out.setdefault("status", "ok")
        out.setdefault("collected_at", collected_at)
        return out
    return {"status": "ok", "collected_at": collected_at, "data": payload}


def _host_info_markdown(path: Path | None) -> list[str]:
    lines = ["## Host PC Info\n\n"]
    if path is None or not path.is_file():
        lines.append("Host PC info is not available.\n\n")
        return lines
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        lines.append(f"Failed to read `{path.name}`: {e}\n\n")
        return lines

    rows = _flatten_json(payload)
    if not rows:
        lines.append("Host PC info is empty.\n\n")
        return lines
    lines.append(f"Source: `{path.name}`\n\n")
    lines.append("| Field | Value |\n")
    lines.append("| --- | --- |\n")
    for key, value in rows:
        lines.append(f"| `{_escape_markdown(key)}` | {_escape_markdown(value)} |\n")
    lines.append("\n")
    return lines


def _plots_markdown(base_dir: Path, plot_paths: Sequence[Path]) -> list[str]:
    lines = ["## Plots\n\n"]
    existing = [path for path in plot_paths if path.is_file()]
    if not existing:
        lines.append("No plot PNG files were generated.\n\n")
        return lines
    for path in existing:
        rel = path.relative_to(base_dir) if path.is_relative_to(base_dir) else path
        title = path.stem.replace("_", " ").title()
        lines.append(f"### {title}\n\n")
        lines.append(f"![{title}]({rel.as_posix()})\n\n")
    return lines


def _table_markdown(path: Path | None) -> list[str]:
    lines = ["## Results Table\n\n"]
    if path is None or not path.is_file():
        lines.append("Results table is not available.\n")
        return lines
    lines.append(path.read_text(encoding="utf-8"))
    if not lines[-1].endswith("\n"):
        lines.append("\n")
    return lines


def _flatten_json(value: Any, *, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, str]] = []
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_json(child, prefix=child_key))
        return rows
    if isinstance(value, list):
        if all(not isinstance(item, (Mapping, list)) for item in value):
            return [(prefix, ", ".join(_scalar_to_text(item) for item in value))]
        rows = []
        for idx, child in enumerate(value):
            rows.extend(_flatten_json(child, prefix=f"{prefix}[{idx}]"))
        return rows
    return [(prefix, _scalar_to_text(value))] if prefix else []


def _scalar_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")