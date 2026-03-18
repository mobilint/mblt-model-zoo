import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ModelMetrics:
    prefill_tps: dict[int, float]
    decode_tps: dict[int, float]
    prefill_tokens_per_j: Optional[float]
    decode_tokens_per_j: Optional[float]
    prefill_j_per_token: Optional[float]
    decode_j_per_token: Optional[float]


def _extract_model_id_from_filename(path: Path) -> Optional[str]:
    if path.suffix.lower() != ".json":
        return None
    stem = path.stem
    if "__" not in stem:
        return None
    _, model_id = stem.split("__", 1)
    if not model_id:
        return None
    return model_id


def _load_model_metrics(path: Path) -> ModelMetrics:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    benchmark = payload.get("benchmark", payload)
    power = payload.get("power", {})
    if not isinstance(power, dict):
        power = {}

    prefill = benchmark.get("prefill_sweep", {})
    decode = benchmark.get("decode_sweep", {})

    prefill_x = prefill.get("x_values", [])
    prefill_tps = prefill.get("tps_values", [])
    decode_x = decode.get("x_values", [])
    decode_tps = decode.get("tps_values", [])

    prefill_map: dict[int, float] = {}
    decode_map: dict[int, float] = {}
    for token, tps in zip(prefill_x, prefill_tps):
        if isinstance(token, int) and isinstance(tps, (int, float)):
            prefill_map[token] = float(tps)
    for token, tps in zip(decode_x, decode_tps):
        if isinstance(token, int) and isinstance(tps, (int, float)):
            decode_map[token] = float(tps)

    def _as_float(v) -> Optional[float]:
        return float(v) if isinstance(v, (int, float)) else None

    return ModelMetrics(
        prefill_tps=prefill_map,
        decode_tps=decode_map,
        prefill_tokens_per_j=_as_float(power.get("prefill_tok_per_j_last")),
        decode_tokens_per_j=_as_float(power.get("decode_tok_per_j_last")),
        prefill_j_per_token=_as_float(power.get("prefill_j_per_tok_last")),
        decode_j_per_token=_as_float(power.get("decode_j_per_tok_last")),
    )


def _collect_folder_metrics(folder: Path) -> dict[str, ModelMetrics]:
    metrics: dict[str, ModelMetrics] = {}
    for path in sorted(folder.glob("*.json")):
        model_id = _extract_model_id_from_filename(path)
        if model_id is None:
            continue
        if model_id in metrics:
            continue
        try:
            metrics[model_id] = _load_model_metrics(path)
        except Exception as e:
            print(f"Warning: failed to parse {path}: {e}")
    return metrics


def _unique_folder_labels(folder_a: Path, folder_b: Path) -> tuple[str, str]:
    a = folder_a.name or str(folder_a)
    b = folder_b.name or str(folder_b)
    if a != b:
        return a, b
    return f"{a} (A)", f"{b} (B)"


def _plot_tps_chart(
    *,
    models: list[str],
    folder_labels: tuple[str, str],
    metrics_a: dict[str, ModelMetrics],
    metrics_b: dict[str, ModelMetrics],
    phase: str,
    output_path: Path,
) -> None:
    if phase not in ("prefill", "decode"):
        raise ValueError("phase must be prefill or decode")

    token_set: set[int] = set()
    for m in models:
        token_set.update((metrics_a[m].prefill_tps if phase == "prefill" else metrics_a[m].decode_tps).keys())
        token_set.update((metrics_b[m].prefill_tps if phase == "prefill" else metrics_b[m].decode_tps).keys())
    tokens = sorted(token_set)

    categories: list[tuple[int, int]] = []
    for token in tokens:
        categories.append((0, token))
        categories.append((1, token))
    if not categories:
        print(f"Skipping {output_path.name}: no {phase} TPS data.")
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
            metric = metrics_a[model] if folder_idx == 0 else metrics_b[model]
            series = metric.prefill_tps if phase == "prefill" else metric.decode_tps
            value = series.get(token)
            if value is None:
                continue
            x_vals.append(value)
            y_vals.append(y[i] + start + c_idx * bar_h)
        if not x_vals:
            continue
        label = f"{folder_labels[folder_idx]} token={token}"
        ax.barh(y_vals, x_vals, height=bar_h * 0.95, label=label, color=cmap(c_idx % 20))

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("TPS (tokens/sec)")
    ax.set_ylabel("model_id")
    ax.set_title(f"{phase.capitalize()} TPS")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_scalar_chart(
    *,
    models: list[str],
    folder_labels: tuple[str, str],
    metrics_a: dict[str, ModelMetrics],
    metrics_b: dict[str, ModelMetrics],
    title: str,
    x_label: str,
    output_path: Path,
    selector: str,
) -> None:
    y = np.arange(len(models), dtype=float)
    bar_h = 0.36

    fig_h = max(5.0, 0.45 * len(models) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    def _get(metric: ModelMetrics) -> Optional[float]:
        return getattr(metric, selector)

    for idx, (label, source, color) in enumerate(
        [
            (folder_labels[0], metrics_a, "#1f77b4"),
            (folder_labels[1], metrics_b, "#ff7f0e"),
        ]
    ):
        x_vals = []
        y_vals = []
        for i, model in enumerate(models):
            value = _get(source[model])
            if value is None:
                continue
            x_vals.append(value)
            y_vals.append(y[i] + (idx - 0.5) * bar_h)
        if x_vals:
            ax.barh(y_vals, x_vals, height=bar_h * 0.95, label=label, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel("model_id")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two benchmark result folders and generate 6 model-wise bar charts."
        )
    )
    parser.add_argument("folder_a", help="first benchmark result folder (relative/absolute)")
    parser.add_argument("folder_b", help="second benchmark result folder (relative/absolute)")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="directory to save PNG charts (default: current directory)",
    )
    args = parser.parse_args()

    folder_a = Path(args.folder_a).expanduser().resolve()
    folder_b = Path(args.folder_b).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not folder_a.is_dir():
        raise SystemExit(f"Not a directory: {folder_a}")
    if not folder_b.is_dir():
        raise SystemExit(f"Not a directory: {folder_b}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_a = _collect_folder_metrics(folder_a)
    metrics_b = _collect_folder_metrics(folder_b)

    common_models = sorted(set(metrics_a.keys()) & set(metrics_b.keys()))
    print(f"Folder A models: {len(metrics_a)}")
    print(f"Folder B models: {len(metrics_b)}")
    print(f"Common models: {len(common_models)}")
    for model in common_models:
        print(f" - {model}")

    if not common_models:
        raise SystemExit("No common model_id found across the two folders.")

    labels = _unique_folder_labels(folder_a, folder_b)

    _plot_tps_chart(
        models=common_models,
        folder_labels=labels,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        phase="prefill",
        output_path=output_dir / "prefill_tps.png",
    )
    _plot_tps_chart(
        models=common_models,
        folder_labels=labels,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        phase="decode",
        output_path=output_dir / "decode_tps.png",
    )
    _plot_scalar_chart(
        models=common_models,
        folder_labels=labels,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        title="Prefill Tokens/J",
        x_label="Tokens/J",
        output_path=output_dir / "prefill_tokens_per_j.png",
        selector="prefill_tokens_per_j",
    )
    _plot_scalar_chart(
        models=common_models,
        folder_labels=labels,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        title="Decode Tokens/J",
        x_label="Tokens/J",
        output_path=output_dir / "decode_tokens_per_j.png",
        selector="decode_tokens_per_j",
    )
    _plot_scalar_chart(
        models=common_models,
        folder_labels=labels,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        title="Prefill J/Token",
        x_label="J/Token",
        output_path=output_dir / "prefill_j_per_token.png",
        selector="prefill_j_per_token",
    )
    _plot_scalar_chart(
        models=common_models,
        folder_labels=labels,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        title="Decode J/Token",
        x_label="J/Token",
        output_path=output_dir / "decode_j_per_token.png",
        selector="decode_j_per_token",
    )

    print(f"Saved charts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
