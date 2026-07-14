"""Compare standardized vision benchmark result directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.common.summary_utils import read_csv_rows


class BenchmarkScore(NamedTuple):
    """One successful standardized vision benchmark score."""

    metric: str
    value: float


def _resolve_results_csv(raw: str) -> Path:
    """Resolve a standard results CSV file from a path argument.

    Args:
        raw: A results directory or explicit CSV path.

    Returns:
        Resolved CSV path.

    Raises:
        ValueError: If the path is not a standard benchmark results file.
    """
    path = Path(raw).expanduser().resolve()
    if path.is_dir():
        path = path / "results.csv"
    if not path.is_file() or path.name != "results.csv":
        raise ValueError(f"Expected a vision benchmark results.csv file or containing directory: {path}")
    return path


def _collect_scores(path: Path) -> dict[str, BenchmarkScore]:
    """Read successful model/core-mode scores from one results CSV.

    Args:
        path: Standardized benchmark ``results.csv`` path.

    Returns:
        Scores keyed by model and core mode.

    Raises:
        ValueError: If a successful result has invalid standardized fields.
    """
    scores: dict[str, BenchmarkScore] = {}
    for row in read_csv_rows(path):
        if row.get("status") != "ok":
            continue
        model = (row.get("model") or "").strip()
        core_mode = (row.get("core_mode") or "").strip()
        metric = (row.get("score_name") or "").strip()
        if not model or not core_mode or not metric:
            raise ValueError(f"Missing model, core_mode, or score_name in {path}.")
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid score for {model}@{core_mode} in {path}.") from exc
        scores.setdefault(f"{model}@{core_mode}", BenchmarkScore(metric=metric, value=score))
    return scores


def _common_targets(scores_by_source: Sequence[dict[str, BenchmarkScore]]) -> list[str]:
    """Return model/core-mode targets shared by all comparison sources.

    Args:
        scores_by_source: Scores parsed from each result source.

    Returns:
        Sorted shared target labels.
    """
    if not scores_by_source:
        return []
    return sorted(set.intersection(*(set(scores) for scores in scores_by_source)))


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a grouped score chart from standardized vision result files.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Zero on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="At least two result directories or results.csv files.")
    parser.add_argument("--output-dir", type=Path, help="Directory for the grouped comparison chart.")
    args = parser.parse_args(argv)
    if len(args.inputs) < 2:
        parser.error("Provide at least two benchmark result sources.")
    try:
        sources = [_resolve_results_csv(raw) for raw in args.inputs]
        scores_by_source = [_collect_scores(source) for source in sources]
    except ValueError as exc:
        parser.error(str(exc))

    targets = _common_targets(scores_by_source)
    if not targets:
        raise SystemExit("No successful model/core-mode target is shared by all input sources.")
    metric_names = {score.metric for scores in scores_by_source for score in scores.values()}
    if len(metric_names) != 1:
        raise SystemExit(f"Inputs contain incompatible benchmark metrics: {', '.join(sorted(metric_names))}.")
    metric_name = next(iter(metric_names))
    from benchmark.common.chart_utils import default_charts_dir, plot_grouped_scalar_barh, source_labels

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_charts_dir(Path(__file__).resolve().parent, sources, use_stem=False)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_grouped_scalar_barh(
        models=targets,
        group_labels=source_labels(sources, use_stem=False),
        grouped_values=[{target: scores[target].value for target in targets} for scores in scores_by_source],
        x_label=metric_name,
        y_label="model@core_mode",
        title=f"Vision benchmark comparison: {metric_name}",
        output_path=output_dir / "score.png",
    )
    print(f"Compared {len(targets)} shared targets using {metric_name}.")
    print(f"Saved chart to: {output_dir / 'score.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
