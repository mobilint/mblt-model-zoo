"""Run reproducible multi-model vision accuracy benchmarks.

This entry point follows the artifact contract used by the Transformers benchmark:
one results directory contains machine-readable JSON and CSV output, an optional
summary, and an accuracy chart.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.common.argparse_utils import parse_positive_int
from benchmark.common.io_utils import safe_filename, write_csv, write_json
from benchmark.common.summary_utils import collect_host_pc_info, markdown_table, write_summary_markdown
from mblt_model_zoo.utils.core_mode import CoreMode, normalize_core_mode

CORE_MODES: tuple[CoreMode, ...] = ("single", "multi", "global4", "global8")
TASK_CHOICES = (
    "image_classification",
    "object_detection",
    "instance_segmentation",
    "pose_estimation",
    "face_detection",
    "obb",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the standardized vision benchmark.

    Args:
        argv: Optional argument sequence. ``None`` reads process arguments.

    Returns:
        Parsed benchmark options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Vision model classes to benchmark.")
    parser.add_argument("--task", choices=TASK_CHOICES, required=True, help="Task shared by all requested models.")
    parser.add_argument("--model-type", default="DEFAULT", help="Model variant from the YAML configuration.")
    parser.add_argument("--model-path", default="", help="Optional local MXQ or ONNX model path for one target.")
    parser.add_argument("--mxq-path", default="", help="Compatibility alias for a local MXQ path.")
    parser.add_argument("--onnx-path", default="", help="Optional local ONNX path for one target.")
    parser.add_argument("--framework", choices=["mxq", "onnx"], help="Explicit inference framework.")
    parser.add_argument(
        "--core-mode",
        default="global8",
        choices=[*CORE_MODES, "all"],
        help="NPU core mode, or `all` to run every supported mode.",
    )
    parser.add_argument("--dev-no", type=int, default=0, help="NPU device number.")
    parser.add_argument("--batch-size", type=parse_positive_int, default=1, help="Validation batch size.")
    parser.add_argument("--data-path", required=True, help="Path to an organized validation dataset.")
    parser.add_argument("--conf-thres", type=float, default=None, help="Optional confidence threshold override.")
    parser.add_argument("--iou-thres", type=float, default=None, help="Optional IoU threshold override.")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark/vision/results"), help="Output directory.")
    parser.add_argument("--no-plot", action="store_true", help="Do not write an accuracy chart.")
    parser.add_argument("--collect-host-info", action="store_true", help="Collect host metadata with mblt-tracker.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed model run.")
    args = parser.parse_args(argv)
    local_paths = [args.model_path, args.mxq_path, args.onnx_path]
    if len(args.models) != 1 and any(local_paths):
        parser.error("--model-path, --mxq-path, and --onnx-path require exactly one --models target.")
    return args


def _core_modes(core_mode: str, framework: str | None = None) -> tuple[str, ...]:
    """Expand the multi-run core-mode shorthand.

    Args:
        core_mode: Requested core mode.
        framework: Explicit inference framework, when provided.

    Returns:
        One or more concrete core modes.
    """
    if framework == "onnx":
        return ("onnx",)
    return CORE_MODES if core_mode == "all" else (normalize_core_mode(core_mode),)


def _evaluate(model: Any, args: argparse.Namespace, run_dir: Path) -> tuple[float, str, dict[str, float]]:
    """Evaluate one model and normalize task metrics for benchmark artifacts.

    Args:
        model: Initialized vision engine.
        args: Parsed benchmark options.
        run_dir: Per-run output directory.

    Returns:
        Primary score, its name, and all normalized metrics.

    Raises:
        ValueError: If the model task differs from the requested benchmark task.
    """
    from mblt_model_zoo.vision.utils.evaluation import eval_coco, eval_dota, eval_imagenet, eval_widerface

    model_task = str(model.post_cfg.get("task", "")).lower()
    if model_task != args.task:
        raise ValueError(f"Model task '{model_task}' does not match requested task '{args.task}'.")
    if args.task == "image_classification":
        result = eval_imagenet(model, args.data_path, args.batch_size)
        return (
            float(result.primary_score),
            "top1_accuracy",
            {
                "top1_accuracy": float(result.top1),
                "top5_accuracy": float(result.top5),
            },
        )
    if args.task in {"object_detection", "instance_segmentation", "pose_estimation"}:
        score = float(eval_coco(model, args.data_path, args.batch_size, args.conf_thres, args.iou_thres))
        return score, "map50_95", {"map50_95": score}
    if args.task == "obb":
        result = eval_dota(
            model,
            args.data_path,
            args.batch_size,
            args.conf_thres,
            args.iou_thres,
            str(run_dir / "dota_task1"),
        )
        return (
            float(result.primary_score),
            "map50_95",
            {
                "map50_95": float(result.map5095),
                "map50": float(result.map50),
            },
        )
    if args.task == "face_detection":
        result = eval_widerface(model, args.data_path, args.batch_size, args.conf_thres, args.iou_thres)
        return (
            float(result.mean_ap),
            "mean_ap",
            {
                "easy_ap": float(result.easy_ap),
                "medium_ap": float(result.medium_ap),
                "hard_ap": float(result.hard_ap),
                "mean_ap": float(result.mean_ap),
            },
        )
    raise ValueError(f"Unsupported vision benchmark task: {args.task}")


def _run_target(model_name: str, core_mode: str, args: argparse.Namespace, results_dir: Path) -> dict[str, Any]:
    """Run and record one model/core-mode benchmark target.

    Args:
        model_name: Vision model class name.
        core_mode: Concrete NPU core mode or the neutral ONNX runtime label.
        args: Parsed benchmark options.
        results_dir: Root directory for benchmark artifacts.

    Returns:
        A normalized benchmark result row.
    """
    from mblt_model_zoo.vision import MBLT_Engine

    label = f"{model_name}@{core_mode}"
    run_dir = results_dir / "runs" / safe_filename(label)
    run_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "model": model_name,
        "core_mode": core_mode,
        "task": args.task,
        "batch_size": args.batch_size,
        "status": "error",
    }
    model = None
    started = time.perf_counter()
    try:
        engine_kwargs: dict[str, Any] = {
            "model_cls": model_name,
            "model_type": args.model_type,
            "model_path": args.model_path,
            "mxq_path": args.mxq_path,
            "onnx_path": args.onnx_path,
            "framework": args.framework,
            "dev_no": args.dev_no,
        }
        if core_mode != "onnx":
            engine_kwargs["core_mode"] = core_mode
        model = MBLT_Engine(
            **engine_kwargs,
        )
        score, score_name, metrics = _evaluate(model, args, run_dir)
        row.update({"status": "ok", "score": score, "score_name": score_name, **metrics})
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["elapsed_s"] = round(time.perf_counter() - started, 6)
        if model is not None:
            model.dispose()
    return row


def _write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace, results_dir: Path) -> None:
    """Write the shared JSON, CSV, chart, and Markdown benchmark artifacts.

    Args:
        rows: Normalized benchmark rows.
        args: Parsed benchmark options.
        results_dir: Destination directory.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "benchmark": "vision", "task": args.task, "results": rows}
    write_json(results_dir / "results.json", payload)
    write_csv(results_dir / "results.csv", rows)

    successful = [row for row in rows if row["status"] == "ok"]
    plot_paths: list[Path] = []
    if successful and not args.no_plot:
        from benchmark.common.chart_utils import plot_simple_barh

        chart_path = results_dir / "accuracy.png"
        plot_simple_barh(
            labels=[f"{row['model']} ({row['core_mode']})" for row in successful],
            values=[float(row["score"]) for row in successful],
            x_label="Accuracy score",
            title=f"{args.task} benchmark accuracy",
            output_path=chart_path,
        )
        plot_paths.append(chart_path)

    host_info_path = collect_host_pc_info(results_dir) if args.collect_host_info else None
    table = markdown_table(
        ["Model", "Core mode", "Metric", "Score", "Elapsed (s)", "Status"],
        [
            [
                row["model"],
                row["core_mode"],
                row.get("score_name", "-"),
                f"{float(row['score']):.5f}" if row.get("score") is not None else "-",
                row["elapsed_s"],
                row["status"],
            ]
            for row in rows
        ],
    )
    table_path = results_dir / "results.md"
    table_path.write_text(table, encoding="utf-8")
    write_summary_markdown(
        results_dir / "summary.md",
        title=f"Vision benchmark: {args.task}",
        host_info_path=host_info_path,
        table_markdown_path=table_path,
        plot_paths=plot_paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standardized vision benchmark.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Zero when every target succeeds, otherwise one.
    """
    args = _parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        for core_mode in _core_modes(args.core_mode, args.framework):
            print(f"Benchmarking {model_name} with core mode {core_mode}...")
            row = _run_target(model_name, core_mode, args, results_dir)
            rows.append(row)
            if row["status"] == "ok":
                print(f"  {row['score_name']}: {row['score']:.5f}")
            else:
                print(f"  failed: {row['error']}")
                if args.fail_fast:
                    _write_outputs(rows, args, results_dir)
                    return 1
    _write_outputs(rows, args, results_dir)
    print(f"Saved benchmark artifacts to: {results_dir}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
