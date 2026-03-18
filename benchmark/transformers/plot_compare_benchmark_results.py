import argparse
import json
import re
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


def _unique_folder_labels(folders: list[Path]) -> list[str]:
    base_labels = [folder.name or str(folder) for folder in folders]
    counts: dict[str, int] = {}
    for label in base_labels:
        counts[label] = counts.get(label, 0) + 1
    seen: dict[str, int] = {}
    out: list[str] = []
    for idx, label in enumerate(base_labels):
        if counts[label] == 1:
            out.append(label)
            continue
        seen[label] = seen.get(label, 0) + 1
        out.append(f"{label} [{seen[label]}/{counts[label]}]")
    if len(set(out)) != len(out):
        out = [f"{label}#{idx + 1}" for idx, label in enumerate(base_labels)]
    return out


def _sanitize_dirname(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return cleaned or "unnamed"


def _default_output_dir(folders: list[Path]) -> Path:
    base = Path(__file__).resolve().parent / "results" / "compare"
    parts: list[str] = []
    for folder in folders:
        name = folder.name or str(folder)
        parts.append(_sanitize_dirname(name))
    return base / "_".join(parts)


def _plot_tps_chart(
    *,
    models: list[str],
    folder_labels: list[str],
    metrics_by_folder: list[dict[str, ModelMetrics]],
    phase: str,
    output_path: Path,
) -> None:
    if phase not in ("prefill", "decode"):
        raise ValueError("phase must be prefill or decode")

    token_set: set[int] = set()
    for m in models:
        for folder_metrics in metrics_by_folder:
            token_set.update(
                (
                    folder_metrics[m].prefill_tps
                    if phase == "prefill"
                    else folder_metrics[m].decode_tps
                ).keys()
            )
    tokens = sorted(token_set)

    categories: list[tuple[int, int]] = []
    for folder_idx in range(len(metrics_by_folder)):
        for token in tokens:
            categories.append((folder_idx, token))
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
            metric = metrics_by_folder[folder_idx][model]
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
    folder_labels: list[str],
    metrics_by_folder: list[dict[str, ModelMetrics]],
    title: str,
    x_label: str,
    output_path: Path,
    selector: str,
) -> None:
    y = np.arange(len(models), dtype=float)
    group_height = 0.82
    bar_h = group_height / max(len(metrics_by_folder), 1)
    start = -group_height / 2 + bar_h / 2

    fig_h = max(5.0, 0.45 * len(models) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    def _get(metric: ModelMetrics) -> Optional[float]:
        return getattr(metric, selector)

    cmap = plt.get_cmap("tab10")
    for idx, (label, source) in enumerate(zip(folder_labels, metrics_by_folder)):
        x_vals = []
        y_vals = []
        for i, model in enumerate(models):
            value = _get(source[model])
            if value is None:
                continue
            x_vals.append(value)
            y_vals.append(y[i] + start + idx * bar_h)
        if x_vals:
            ax.barh(
                y_vals,
                x_vals,
                height=bar_h * 0.95,
                label=label,
                color=cmap(idx % 10),
            )

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
            "Compare N benchmark result folders and generate 6 model-wise bar charts."
        )
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="benchmark result folders (relative/absolute). Pass 2 or more.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "directory to save PNG charts (default: "
            "benchmark/transformers/results/compare/<dir1_dir2_...>)"
        ),
    )
    args = parser.parse_args()

    folders = [Path(folder).expanduser().resolve() for folder in args.folders]
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(folders)
    )

    if len(folders) < 2:
        raise SystemExit("Please provide at least 2 folders.")
    for folder in folders:
        if not folder.is_dir():
            raise SystemExit(f"Not a directory: {folder}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_folder = [_collect_folder_metrics(folder) for folder in folders]
    folder_labels = _unique_folder_labels(folders)

    model_sets = [set(metrics.keys()) for metrics in metrics_by_folder]
    common_models = sorted(set.intersection(*model_sets))
    for label, folder, metrics in zip(folder_labels, folders, metrics_by_folder):
        print(f"Folder {label}: {folder} -> {len(metrics)} models")
    print(f"Common models across all folders: {len(common_models)}")
    for model in common_models:
        print(f" - {model}")

    if not common_models:
        raise SystemExit("No common model_id found across all input folders.")

    _plot_tps_chart(
        models=common_models,
        folder_labels=folder_labels,
        metrics_by_folder=metrics_by_folder,
        phase="prefill",
        output_path=output_dir / "prefill_tps.png",
    )
    _plot_tps_chart(
        models=common_models,
        folder_labels=folder_labels,
        metrics_by_folder=metrics_by_folder,
        phase="decode",
        output_path=output_dir / "decode_tps.png",
    )
    _plot_scalar_chart(
        models=common_models,
        folder_labels=folder_labels,
        metrics_by_folder=metrics_by_folder,
        title="Prefill Tokens/J",
        x_label="Tokens/J",
        output_path=output_dir / "prefill_tokens_per_j.png",
        selector="prefill_tokens_per_j",
    )
    _plot_scalar_chart(
        models=common_models,
        folder_labels=folder_labels,
        metrics_by_folder=metrics_by_folder,
        title="Decode Tokens/J",
        x_label="Tokens/J",
        output_path=output_dir / "decode_tokens_per_j.png",
        selector="decode_tokens_per_j",
    )
    _plot_scalar_chart(
        models=common_models,
        folder_labels=folder_labels,
        metrics_by_folder=metrics_by_folder,
        title="Prefill J/Token",
        x_label="J/Token",
        output_path=output_dir / "prefill_j_per_token.png",
        selector="prefill_j_per_token",
    )
    _plot_scalar_chart(
        models=common_models,
        folder_labels=folder_labels,
        metrics_by_folder=metrics_by_folder,
        title="Decode J/Token",
        x_label="J/Token",
        output_path=output_dir / "decode_j_per_token.png",
        selector="decode_j_per_token",
    )

    print(f"Saved charts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
