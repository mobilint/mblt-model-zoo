import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from benchmark.common.chart_utils import (
        default_charts_dir as _default_charts_dir_common,
    )
    from benchmark.common.chart_utils import (
        plot_grouped_scalar_barh,
    )
    from benchmark.common.chart_utils import (
        source_labels as _source_labels_common,
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
    _source_labels_common = _common_mod.source_labels


@dataclass
class VisionMetrics:
    mAP: Optional[float]
    fps: Optional[float]
    power_w: Optional[float]
    fps_per_w: Optional[float]
    util_pct: Optional[float]


def default_charts_dir(script_dir: Path, sources: list[Path]) -> Path:
    return _default_charts_dir_common(script_dir, sources, use_stem=True)


def source_labels(sources: list[Path]) -> list[str]:
    return _source_labels_common(sources, use_stem=True)


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
    raise ValueError(f"Multiple CSV files found in directory: {path}. Please pass explicit CSV file paths.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Compare N vision benchmark CSV files and generate model-wise bar charts.")
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files (or directories that contain one CSV file). Pass 2 or more.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=("directory to save PNG charts (default: benchmark/vision/results/charts/<input1_input2_...>)"),
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
        Path(args.output_dir).expanduser().resolve() if args.output_dir else default_charts_dir(script_dir, sources)
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

    plot_specs = [
        ("mAP", "mAP", "mAP.png", lambda m: m.mAP),
        ("FPS", "FPS", "fps.png", lambda m: m.fps),
        ("Power", "Power (W)", "power_w.png", lambda m: m.power_w),
        ("FPS/W", "FPS/W", "fps_per_w.png", lambda m: m.fps_per_w),
        ("Utilization", "Utilization (%)", "util_pct.png", lambda m: m.util_pct),
    ]
    for title, x_label, file_name, selector in plot_specs:
        grouped_values = [{model: selector(source[model]) for model in models} for source in metrics_by_source]
        plot_grouped_scalar_barh(
            models=models,
            group_labels=labels,
            grouped_values=grouped_values,
            x_label=x_label,
            y_label="model",
            title=title,
            output_path=output_dir / file_name,
            fig_width=14.0,
        )

    print(f"Saved charts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
