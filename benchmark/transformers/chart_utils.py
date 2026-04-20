import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from benchmark.common.chart_utils import (
        default_charts_dir as _default_charts_dir_common,
    )
    from benchmark.common.chart_utils import (
        plot_grouped_scalar_barh,
    )
    from benchmark.common.chart_utils import (
        sanitize_text as _sanitize_text_common,
    )
    from benchmark.common.chart_utils import (
        source_labels as _source_labels_common,
    )
    from benchmark.common.chart_utils import (
        source_prefix as _source_prefix_common,
    )
except ModuleNotFoundError:
    _common_path = Path(__file__).resolve().parents[1] / "common" / "chart_utils.py"
    _spec = importlib.util.spec_from_file_location("benchmark_common_chart_utils", _common_path)
    if _spec is None or _spec.loader is None:
        raise
    _common_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_common_mod)
    _default_charts_dir_common = _common_mod.default_charts_dir
    plot_grouped_scalar_barh = _common_mod.plot_grouped_scalar_barh
    _sanitize_text_common = _common_mod.sanitize_text
    _source_labels_common = _common_mod.source_labels
    _source_prefix_common = _common_mod.source_prefix


@dataclass
class ModelMetrics:
    prefill_tps: dict[int, float]
    decode_tps: dict[int, float]
    prefill_latency_ms: dict[int, float]
    decode_duration_ms: dict[int, float]
    prefill_tokens_per_j: Optional[float]
    decode_tokens_per_j: Optional[float]
    prefill_j_per_token: Optional[float]
    decode_j_per_token: Optional[float]
    avg_power_w: Optional[float]
    total_energy_j: Optional[float]
    avg_utilization_pct: Optional[float]
    p99_utilization_pct: Optional[float]
    avg_temperature_c: Optional[float]
    p99_temperature_c: Optional[float]
    avg_memory_used_mb: Optional[float]
    p99_memory_used_mb: Optional[float]
    avg_memory_used_pct: Optional[float]
    p99_memory_used_pct: Optional[float]


def sanitize_text(text: str) -> str:
    return _sanitize_text_common(text)


def folder_prefix(folders: list[Path]) -> str:
    return _source_prefix_common(folders, use_stem=False)


def default_charts_dir(script_dir: Path, folders: list[Path]) -> Path:
    return _default_charts_dir_common(script_dir, folders, use_stem=False)


def folder_labels(folders: list[Path]) -> list[str]:
    return _source_labels_common(folders, use_stem=False)


def _extract_model_id(path: Path, payload: dict) -> Optional[str]:
    model_in_payload = payload.get("model")
    if isinstance(model_in_payload, str) and model_in_payload:
        return model_in_payload
    if path.suffix.lower() != ".json":
        return None
    stem = path.stem
    if "__" not in stem:
        return None
    # Backward compatibility: group_id__model_id
    _, model_id = stem.split("__", 1)
    return model_id or None


def _as_float(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None


def load_model_metrics(path: Path) -> Optional[tuple[str, ModelMetrics]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    model_id = _extract_model_id(path, payload)
    if not model_id:
        return None

    benchmark = payload.get("benchmark", payload)
    device = payload.get("device", {})
    if not isinstance(device, dict):
        device = {}

    prefill = benchmark.get("prefill_sweep", {})
    decode = benchmark.get("decode_sweep", {})
    prefill_x = prefill.get("x_values", [])
    prefill_tps = prefill.get("tps_values", [])
    prefill_time = prefill.get("time_values", [])
    decode_x = decode.get("x_values", [])
    decode_tps = decode.get("tps_values", [])
    decode_time = decode.get("time_values", [])

    prefill_tps_map: dict[int, float] = {}
    decode_tps_map: dict[int, float] = {}
    prefill_latency_map: dict[int, float] = {}
    decode_duration_map: dict[int, float] = {}
    for token, tps, t in zip(prefill_x, prefill_tps, prefill_time):
        if isinstance(token, int):
            if isinstance(tps, (int, float)):
                prefill_tps_map[token] = float(tps)
            if isinstance(t, (int, float)):
                prefill_latency_map[token] = float(t) * 1000.0
    for token, tps, t in zip(decode_x, decode_tps, decode_time):
        if isinstance(token, int):
            if isinstance(tps, (int, float)):
                decode_tps_map[token] = float(tps)
            if isinstance(t, (int, float)):
                decode_duration_map[token] = float(t) * 1000.0

    return model_id, ModelMetrics(
        prefill_tps=prefill_tps_map,
        decode_tps=decode_tps_map,
        prefill_latency_ms=prefill_latency_map,
        decode_duration_ms=decode_duration_map,
        prefill_tokens_per_j=_as_float(device.get("prefill_tok_per_j_last")),
        decode_tokens_per_j=_as_float(device.get("decode_tok_per_j_last")),
        prefill_j_per_token=_as_float(device.get("prefill_j_per_tok_last")),
        decode_j_per_token=_as_float(device.get("decode_j_per_tok_last")),
        avg_power_w=_as_float(device.get("avg_power_w")),
        total_energy_j=_as_float(device.get("total_energy_j")),
        avg_utilization_pct=_as_float(device.get("avg_utilization_pct")),
        p99_utilization_pct=_as_float(device.get("p99_utilization_pct")),
        avg_temperature_c=_as_float(device.get("avg_temperature_c")),
        p99_temperature_c=_as_float(device.get("p99_temperature_c")),
        avg_memory_used_mb=_as_float(device.get("avg_memory_used_mb")),
        p99_memory_used_mb=_as_float(device.get("p99_memory_used_mb")),
        avg_memory_used_pct=_as_float(device.get("avg_memory_used_pct")),
        p99_memory_used_pct=_as_float(device.get("p99_memory_used_pct")),
    )


def collect_folder_metrics(folder: Path) -> dict[str, ModelMetrics]:
    metrics: dict[str, ModelMetrics] = {}
    for path in sorted(folder.glob("*.json")):
        try:
            loaded = load_model_metrics(path)
        except Exception as e:
            print(f"Warning: failed to parse {path}: {e}")
            continue
        if loaded is None:
            continue
        model_id, metric = loaded
        if model_id in metrics:
            continue
        metrics[model_id] = metric
    return metrics


def common_models(metrics_by_folder: list[dict[str, ModelMetrics]]) -> list[str]:
    if not metrics_by_folder:
        return []
    model_sets = [set(metrics.keys()) for metrics in metrics_by_folder]
    if not model_sets:
        return []
    return sorted(set.intersection(*model_sets))


def plot_token_chart(
    *,
    models: list[str],
    folder_labels: list[str],
    metrics_by_folder: list[dict[str, ModelMetrics]],
    token_selector: Callable[[ModelMetrics], dict[int, float]],
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    if not models:
        return
    token_set: set[int] = set()
    for model in models:
        for folder_metrics in metrics_by_folder:
            token_set.update(token_selector(folder_metrics[model]).keys())
    tokens = sorted(token_set)
    categories: list[tuple[int, int]] = []
    for folder_idx in range(len(metrics_by_folder)):
        for token in tokens:
            categories.append((folder_idx, token))
    if not categories:
        return

    y = np.arange(len(models), dtype=float)
    group_height = 0.82
    bar_h = group_height / max(len(categories), 1)
    start = -group_height / 2 + bar_h / 2
    fig_h = max(5.0, 0.45 * len(models) + 2.0)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    cmap = plt.get_cmap("tab20")
    for c_idx, (folder_idx, token) in enumerate(categories):
        x_vals = []
        y_vals = []
        for i, model in enumerate(models):
            value = token_selector(metrics_by_folder[folder_idx][model]).get(token)
            if value is None:
                continue
            x_vals.append(float(value))
            y_vals.append(y[i] + start + c_idx * bar_h)
        if not x_vals:
            continue
        label = f"{folder_labels[folder_idx]} token={token}"
        ax.barh(y_vals, x_vals, height=bar_h * 0.95, label=label, color=cmap(c_idx % 20))
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel("model_id")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_scalar_chart(
    *,
    models: list[str],
    folder_labels: list[str],
    metrics_by_folder: list[dict[str, ModelMetrics]],
    scalar_selector: Callable[[ModelMetrics], Optional[float]],
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    grouped_values = [{model: scalar_selector(source[model]) for model in models} for source in metrics_by_folder]
    plot_grouped_scalar_barh(
        models=models,
        group_labels=folder_labels,
        grouped_values=grouped_values,
        x_label=x_label,
        y_label="model_id",
        title=title,
        output_path=output_path,
        fig_width=14.0,
    )
