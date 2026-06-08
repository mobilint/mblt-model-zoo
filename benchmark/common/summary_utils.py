"""Markdown summary helpers for benchmark outputs."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

HOST_PC_INFO_FILENAME = "host_pc_info.json"

_HOST_INFO_SECTION_ORDER = ("CPU", "Motherboard", "DRAM", "NPU")

_HOST_INFO_SECTION_ALIASES = {
    "cpu": "CPU",
    "processor": "CPU",
    "motherboard": "Motherboard",
    "mainboard": "Motherboard",
    "baseboard": "Motherboard",
    "board": "Motherboard",
    "dram": "DRAM",
    "memory": "DRAM",
    "ram": "DRAM",
    "dimm": "DRAM",
    "npu": "NPU",
    "npus": "NPU",
    "neural": "NPU",
    "accelerator": "NPU",
}

_NPU_ARRAY_KEY_RE = re.compile(r"(?:^|\.)npus\[(\d+)\]\.(.+)$")


_PLOT_TITLES_BY_NAME = {
    "rtf.png": "Real-Time Factor",
    "wer.png": "Word Error Rate",
    "cer.png": "Character Error Rate",
    "prefill_tps.png": "Prefill Tokens Per Second",
    "measure_prefill_tps.png": "Prefill Tokens Per Second",
    "llm_prefill_tps.png": "Prefill Tokens Per Second",
    "measure_llm_prefill_tps.png": "Prefill Tokens Per Second",
    "prefill_tokens_per_j.png": "Prefill Tokens Per Joule",
    "measure_prefill_tokens_per_j.png": "Prefill Tokens Per Joule",
    "llm_prefill_tokens_per_j.png": "Prefill Tokens Per Joule",
    "measure_llm_prefill_tokens_per_j.png": "Prefill Tokens Per Joule",
    "decode_tps.png": "Decode Tokens Per Second",
    "measure_decode_tps.png": "Decode Tokens Per Second",
    "llm_decode_tps.png": "Decode Tokens Per Second",
    "measure_llm_decode_tps.png": "Decode Tokens Per Second",
    "decode_tokens_per_j.png": "Decode Tokens Per Joule",
    "measure_decode_tokens_per_j.png": "Decode Tokens Per Joule",
    "llm_decode_tokens_per_j.png": "Decode Tokens Per Joule",
    "measure_llm_decode_tokens_per_j.png": "Decode Tokens Per Joule",
    "avg_power_w.png": "Power",
    "measure_avg_power_w.png": "Power",
    "avg_temperature_c.png": "Temperature",
    "measure_avg_temperature_c.png": "Temperature",
    "avg_utilization_pct.png": "Utilization",
    "measure_avg_utilization_pct.png": "Utilization",
    "avg_memory_used_mb.png": "Memory Used Megabytes",
    "measure_avg_memory_used_mb.png": "Memory Used Megabytes",
    "total_energy_j.png": "Total Energy",
    "measure_total_energy_j.png": "Total Energy",
}

_PLOT_NAME_ORDER = {name: idx for idx, name in enumerate(_PLOT_TITLES_BY_NAME)}


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
    plot_tables: Mapping[str, str] | None = None,
) -> None:
    """Write a benchmark summary Markdown file with host info, plots, and table.

    Args:
        path: Summary Markdown destination.
        title: Document title.
        host_info_path: JSON file written by :func:`collect_host_pc_info`.
        table_markdown_path: Existing Markdown table to include.
        plot_paths: PNG files to embed in the summary.
        plot_tables: Optional Markdown tables keyed by plot PNG filename. When provided, matching tables are rendered
            directly below each plot and the bottom combined table is omitted.
    """
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    host_info_lines = _host_info_markdown(Path(host_info_path) if host_info_path else None)
    lines = [f"# {title}\n\n"]
    lines.extend(_plots_markdown(summary_path.parent, [Path(p) for p in plot_paths], plot_tables=plot_tables))
    if not plot_tables:
        lines.extend(_table_markdown(Path(table_markdown_path) if table_markdown_path else None))
    lines.extend(host_info_lines)
    summary_path.write_text("".join(lines), encoding="utf-8")


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    """Read CSV rows if the file exists.

    Args:
        path: CSV path to read.

    Returns:
        CSV rows as dictionaries, or an empty list when the file does not exist.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        return []
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Build a compact Markdown table with right-aligned metric columns.

    Args:
        headers: Table headers.
        rows: Table row values.

    Returns:
        Markdown table text, or an empty string for empty rows.
    """
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(_escape_markdown(header) for header in headers) + " |\n",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |\n",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_summary_cell(value) for value in row) + " |\n")
    return "".join(lines)


def scalar_plot_table(rows: Sequence[Mapping[str, Any]], *, value_key: str, unit_header: str) -> str:
    """Build a model/value table for one scalar plot.

    Args:
        rows: Rows containing a ``model`` key and the requested scalar key.
        value_key: Key containing the scalar value.
        unit_header: Header for the value column.

    Returns:
        Markdown table text.
    """
    return markdown_table(["Model", unit_header], [[row.get("model"), row.get(value_key)] for row in rows])


def token_sweep_plot_table(
    models: Sequence[str],
    metrics_by_model: Mapping[str, Any],
    *,
    value_key: str,
) -> str:
    """Build a model/token table for one token-sweep plot.

    Args:
        models: Model names to include.
        metrics_by_model: Mapping from model name to metric object with token dictionaries.
        value_key: Attribute name containing a ``dict[int, float]`` token metric.

    Returns:
        Markdown table text.
    """
    token_set: set[int] = set()
    for model in models:
        token_set.update(getattr(metrics_by_model[model], value_key).keys())
    tokens = sorted(token_set)
    if not tokens:
        return ""
    table_rows = []
    for model in models:
        values = getattr(metrics_by_model[model], value_key)
        table_rows.append([model, *(values.get(token) for token in tokens)])
    return markdown_table(["Model", *(f"{token} tokens" for token in tokens)], table_rows)


def write_token_combined_markdown(
    path: Path | str,
    tps_rows: Sequence[Mapping[str, Any]],
    device_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a wide-form token sweep Markdown table shared by transformer benchmarks.

    Args:
        path: Markdown output path.
        tps_rows: Long-form TPS rows from ``BenchmarkResult.iter_rows``.
        device_rows: Per-model device metric rows.
    """
    if not tps_rows:
        return
    models = sorted({str(r["model"]) for r in tps_rows})
    prefill_tokens = sorted(
        {int(r["tokens"]) for r in tps_rows if str(r.get("phase")) == "prefill" and _is_int_like(r.get("tokens"))}
    )
    decode_tokens = sorted(
        {int(r["tokens"]) for r in tps_rows if str(r.get("phase")) == "decode" and _is_int_like(r.get("tokens"))}
    )
    tps_map: dict[tuple[str, str, int], float] = {}
    time_map: dict[tuple[str, str, int], float] = {}
    npu_pct_map: dict[tuple[str, str, int], float] = {}
    for row in tps_rows:
        model = str(row["model"])
        phase = str(row["phase"])
        token = int(row["tokens"])
        tps_val = row.get("tps")
        time_ms_val = row.get("time_ms")
        npu_pct_val = row.get("avg_npu_token_latency_pct")
        if isinstance(tps_val, (int, float)):
            tps_map[(model, phase, token)] = float(tps_val)
        if isinstance(time_ms_val, (int, float)):
            time_map[(model, phase, token)] = float(time_ms_val)
        if isinstance(npu_pct_val, (int, float)):
            npu_pct_map[(model, phase, token)] = float(npu_pct_val)

    device_map = {str(r["model"]): r for r in device_rows if isinstance(r.get("model"), str)}
    device_cols = [
        "avg_power_w",
        "p99_power_w",
        "avg_utilization_pct",
        "p99_utilization_pct",
        "avg_temperature_c",
        "p99_temperature_c",
        "avg_memory_used_mb",
        "p99_memory_used_mb",
        "total_memory_mb",
        "avg_memory_used_pct",
        "p99_memory_used_pct",
        "total_energy_j",
        "prefill_tps_last",
        "decode_tps_last",
        "prefill_tok_per_j_last",
        "decode_tok_per_j_last",
        "prefill_j_per_tok_last",
        "decode_j_per_tok_last",
    ]

    headers = ["model"]
    headers.extend([f"prefill_tps_{t}" for t in prefill_tokens])
    headers.extend([f"decode_tps_{t}" for t in decode_tokens])
    headers.extend([f"prefill_latency_ms_{t}" for t in prefill_tokens])
    headers.extend([f"decode_duration_ms_{t}" for t in decode_tokens])
    headers.extend([f"prefill_npu_latency_pct_{t}" for t in prefill_tokens])
    headers.extend([f"decode_npu_latency_pct_{t}" for t in decode_tokens])
    headers.extend(device_cols)

    rows: list[list[str]] = []
    for model in models:
        values: list[str] = [model]
        for token in prefill_tokens:
            values.append(_format_optional_float(tps_map.get((model, "prefill", token))))
        for token in decode_tokens:
            values.append(_format_optional_float(tps_map.get((model, "decode", token))))
        for token in prefill_tokens:
            values.append(_format_optional_float(time_map.get((model, "prefill", token))))
        for token in decode_tokens:
            values.append(_format_optional_float(time_map.get((model, "decode", token))))
        for token in prefill_tokens:
            values.append(_format_optional_float(npu_pct_map.get((model, "prefill", token))))
        for token in decode_tokens:
            values.append(_format_optional_float(npu_pct_map.get((model, "decode", token))))

        drow = device_map.get(model, {})
        for col in device_cols:
            v = drow.get(col) if isinstance(drow, Mapping) else None
            values.append(_format_optional_float(v))
        rows.append(values)

    Path(path).write_text(markdown_table(headers, rows), encoding="utf-8")


def existing_png_paths(results_dir: Path | str, *, prefixes: Sequence[str] | None = None) -> list[Path]:
    """Return sorted PNG paths under a benchmark results directory.

    Args:
        results_dir: Directory to scan.
        prefixes: Optional filename prefixes to include.

    Returns:
        Sorted PNG files matching the optional prefixes.
    """
    out_dir = Path(results_dir)
    paths = sorted(
        out_dir.glob("*.png"),
        key=lambda path: (_plot_sort_key(path), path.name),
    )
    if prefixes is None:
        return [path for path in paths if _plot_title(path) is not None]
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

    sections = _host_info_sections(payload)
    if not sections:
        lines.append("Host PC info is empty.\n\n")
        return lines
    lines.append(f"Source: `{path.name}`\n\n")
    for title, rows in sections:
        if title == "NPU":
            _append_npu_info_markdown(lines, rows)
        else:
            _append_host_info_table(lines, title, rows)
    return lines


def _plots_markdown(
    base_dir: Path,
    plot_paths: Sequence[Path],
    *,
    plot_tables: Mapping[str, str] | None = None,
) -> list[str]:
    lines = ["## Plots\n\n"]
    existing = [path for path in plot_paths if path.is_file()]
    if not existing:
        lines.append("No plot PNG files were generated.\n\n")
        return lines
    tables = plot_tables or {}
    for path in existing:
        rel = path.relative_to(base_dir) if path.is_relative_to(base_dir) else path
        title = _plot_title(path) or path.stem.replace("_", " ").title()
        lines.append(f"### {title}\n\n")
        lines.append(f"![{title}]({rel.as_posix()})\n\n")
        table = tables.get(path.name)
        if table:
            lines.append(table)
            if not lines[-1].endswith("\n"):
                lines.append("\n")
            lines.append("\n")
    return lines


def _plot_title(path: Path) -> str | None:
    title = _PLOT_TITLES_BY_NAME.get(path.name)
    if title is not None:
        return title

    stem = path.stem
    if stem.startswith("rtf_beams"):
        return "Real-Time Factor"
    if stem.startswith("wer_beams"):
        return "Word Error Rate"
    if stem.startswith("cer_beams"):
        return "Character Error Rate"
    return None


def _plot_sort_key(path: Path) -> int:
    if path.name in _PLOT_NAME_ORDER:
        return _PLOT_NAME_ORDER[path.name]
    stem = path.stem
    if stem.startswith("rtf_beams"):
        return len(_PLOT_NAME_ORDER)
    if stem.startswith("wer_beams"):
        return len(_PLOT_NAME_ORDER) + 1
    if stem.startswith("cer_beams"):
        return len(_PLOT_NAME_ORDER) + 2
    return len(_PLOT_NAME_ORDER) + 100


def _host_info_sections(payload: Any) -> list[tuple[str, list[tuple[str, str]]]]:
    rows = _flatten_json(payload)
    if not rows:
        return []

    grouped: dict[str, list[tuple[str, str]]] = {section: [] for section in _HOST_INFO_SECTION_ORDER}
    general_rows: list[tuple[str, str]] = []
    for key, value in rows:
        section = _host_info_section_for_key(key)
        if section is None:
            general_rows.append((key, value))
        else:
            grouped[section].append((key, value))

    sections: list[tuple[str, list[tuple[str, str]]]] = []
    if general_rows:
        sections.append(("General", general_rows))
    sections.extend((section, grouped[section]) for section in _HOST_INFO_SECTION_ORDER if grouped[section])
    return sections


def _append_host_info_table(lines: list[str], title: str, rows: Sequence[tuple[str, str]]) -> None:
    """Append one host info section as a Markdown table."""
    lines.append(f"### {title}\n\n")
    _append_field_value_table(lines, rows)


def _append_npu_info_markdown(lines: list[str], rows: Sequence[tuple[str, str]]) -> None:
    """Append NPU host info with ``npus`` array entries grouped by index."""
    common_rows: list[tuple[str, str]] = []
    indexed_rows: dict[int, list[tuple[str, str]]] = {}
    for key, value in rows:
        npu_key = _split_npu_array_key(key)
        if npu_key is None:
            common_rows.append((key, value))
            continue
        index, field = npu_key
        indexed_rows.setdefault(index, []).append((field, value))

    if not indexed_rows:
        _append_host_info_table(lines, "NPU", rows)
        return

    lines.append("### NPU\n\n")
    if common_rows:
        lines.append("#### General\n\n")
        _append_field_value_table(lines, common_rows)
    for index in sorted(indexed_rows):
        lines.append(f"#### NPU {index}\n\n")
        _append_field_value_table(lines, indexed_rows[index])


def _append_field_value_table(lines: list[str], rows: Sequence[tuple[str, str]]) -> None:
    """Append field/value rows as a Markdown table."""
    lines.append("| Field | Value |\n")
    lines.append("| --- | --- |\n")
    for key, value in rows:
        lines.append(f"| `{_escape_markdown(key)}` | {_escape_markdown(value)} |\n")
    lines.append("\n")


def _host_info_section_for_key(key: str) -> str | None:
    if _split_npu_array_key(key) is not None:
        return "NPU"
    normalized = key.replace("_", ".").replace("-", ".").replace(" ", ".").lower()
    parts = [part for part in normalized.replace("[", ".").replace("]", ".").split(".") if part]
    for part in parts:
        section = _HOST_INFO_SECTION_ALIASES.get(part)
        if section is not None:
            return section
    return None


def _split_npu_array_key(key: str) -> tuple[int, str] | None:
    """Return the NPU array index and field for flattened ``npus`` keys."""
    match = _NPU_ARRAY_KEY_RE.search(key)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _host_info_section_title(value: str) -> str:
    upper_names = {"cpu": "CPU", "dram": "DRAM", "gpu": "GPU", "npu": "NPU", "os": "OS"}
    normalized = value.replace("_", " ").replace("-", " ").strip()
    lowered = normalized.lower()
    if lowered in upper_names:
        return upper_names[lowered]
    return " ".join(upper_names.get(part.lower(), part.capitalize()) for part in normalized.split())


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


def _format_summary_cell(value: Any) -> str:
    """Format one benchmark summary table value."""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    if isinstance(value, str):
        try:
            return f"{float(value):.6f}"
        except ValueError:
            return _escape_markdown(value)
    return _escape_markdown(str(value))


def _format_optional_float(value: Any) -> str:
    """Format a numeric value for compact benchmark tables."""
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else ""


def _is_int_like(value: Any) -> bool:
    """Return whether a value can be losslessly parsed as an integer token count."""
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        try:
            int(value)
        except ValueError:
            return False
        return True
    return False


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")
