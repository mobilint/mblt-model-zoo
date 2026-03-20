import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class VisionMetrics:
    mAP: Optional[float]
    fps: Optional[float]
    power_w: Optional[float]
    fps_per_w: Optional[float]
    util_pct: Optional[float]


def sanitize_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return cleaned or "unnamed"


def folder_prefix(sources: list[Path]) -> str:
    return "_".join(sanitize_text(path.stem or path.name) for path in sources)


def default_charts_dir(script_dir: Path, sources: list[Path]) -> Path:
    return script_dir / "results" / "charts" / folder_prefix(sources)


def source_labels(sources: list[Path]) -> list[str]:
    labels = [source.stem or source.name for source in sources]
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    seen: dict[str, int] = {}
    out: list[str] = []
    for idx, label in enumerate(labels):
        if counts[label] == 1:
            out.append(label)
            continue
        seen[label] = seen.get(label, 0) + 1
        out.append(f"{label} [{seen[label]}/{counts[label]}]")

    if len(set(out)) != len(out):
        out = [f"{label}#{idx + 1}" for idx, label in enumerate(labels)]
    return out


def _as_float(raw: str | None) -> Optional[float]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def collect_csv_metrics(path: Path) -> dict[str, VisionMetrics]:
    metrics: dict[str, VisionMetrics] = {}
    rows = _read_csv_rows(path)
    for row in rows:
        model = (row.get("model") or "").strip()
        if not model or model in metrics:
            continue
        metrics[model] = VisionMetrics(
            mAP=_as_float(row.get("mAP")),
            fps=_as_float(row.get("FPS")),
            power_w=_as_float(row.get("Power(W)")),
            fps_per_w=_as_float(row.get("FPS/W")),
            util_pct=_as_float(row.get("Util(%)")),
        )
    return metrics


def common_models(metrics_by_source: list[dict[str, VisionMetrics]]) -> list[str]:
    if not metrics_by_source:
        return []
    model_sets = [set(metrics.keys()) for metrics in metrics_by_source]
    if not model_sets:
        return []
    return sorted(set.intersection(*model_sets))


def _resolve_csv_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Input is not a CSV file: {path}")
        return path

    if not path.is_dir():
        raise ValueError(f"Input is neither CSV nor directory: {path}")

    preferred = path / "results.csv"
    if preferred.is_file():
        return preferred

    csv_files = sorted(path.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise ValueError(f"No CSV found in directory: {path}")
    raise ValueError(
        f"Multiple CSV files found in directory: {path}. "
        "Please pass explicit CSV file paths."
    )


def plot_scalar_chart(
    *,
    models: list[str],
    source_labels_: list[str],
    metrics_by_source: list[dict[str, VisionMetrics]],
    scalar_selector: Callable[[VisionMetrics], Optional[float]],
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    if not models:
        return

    y = np.arange(len(models), dtype=float)
    group_height = 0.82
    bar_h = group_height / max(len(metrics_by_source), 1)
    start = -group_height / 2 + bar_h / 2
    fig_h = max(5.0, 0.45 * len(models) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    cmap = plt.get_cmap("tab10")

    for idx, (label, source) in enumerate(zip(source_labels_, metrics_by_source)):
        x_vals = []
        y_vals = []
        for i, model in enumerate(models):
            value = scalar_selector(source[model])
            if value is None:
                continue
            x_vals.append(float(value))
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
    ax.set_ylabel("model")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare N vision benchmark CSV files and generate model-wise bar charts."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files (or directories that contain one CSV file). Pass 2 or more.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "directory to save PNG charts "
            "(default: benchmark/vision/results/charts/<input1_input2_...>)"
        ),
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        raise SystemExit("Please provide at least 2 CSV inputs.")

    try:
        sources = [_resolve_csv_path(raw) for raw in args.inputs]
    except ValueError as e:
        raise SystemExit(str(e)) from e

    script_dir = Path(__file__).resolve().parent
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_charts_dir(script_dir, sources)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_source = [collect_csv_metrics(path) for path in sources]
    labels = source_labels(sources)
    models = common_models(metrics_by_source)

    for label, source, metrics in zip(labels, sources, metrics_by_source):
        print(f"Source {label}: {source} -> {len(metrics)} models")
    print(f"Common models across all sources: {len(models)}")
    for model in models:
        print(f" - {model}")

    if not models:
        for label, metrics in zip(labels, metrics_by_source):
            sample = sorted(metrics.keys())[:10]
            print(f"[debug] {label} model keys (up to 10): {sample}")
        raise SystemExit("No common model found across all input CSV files.")

    plot_scalar_chart(
        models=models,
        source_labels_=labels,
        metrics_by_source=metrics_by_source,
        scalar_selector=lambda m: m.mAP,
        title="mAP",
        x_label="mAP",
        output_path=output_dir / "mAP.png",
    )
    plot_scalar_chart(
        models=models,
        source_labels_=labels,
        metrics_by_source=metrics_by_source,
        scalar_selector=lambda m: m.fps,
        title="FPS",
        x_label="FPS",
        output_path=output_dir / "fps.png",
    )
    plot_scalar_chart(
        models=models,
        source_labels_=labels,
        metrics_by_source=metrics_by_source,
        scalar_selector=lambda m: m.power_w,
        title="Power",
        x_label="Power (W)",
        output_path=output_dir / "power_w.png",
    )
    plot_scalar_chart(
        models=models,
        source_labels_=labels,
        metrics_by_source=metrics_by_source,
        scalar_selector=lambda m: m.fps_per_w,
        title="FPS/W",
        x_label="FPS/W",
        output_path=output_dir / "fps_per_w.png",
    )
    plot_scalar_chart(
        models=models,
        source_labels_=labels,
        metrics_by_source=metrics_by_source,
        scalar_selector=lambda m: m.util_pct,
        title="Utilization",
        x_label="Utilization (%)",
        output_path=output_dir / "util_pct.png",
    )

    print(f"Saved charts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
