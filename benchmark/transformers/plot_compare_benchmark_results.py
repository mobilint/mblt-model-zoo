import argparse
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping

try:
    from benchmark.transformers.compare_metrics import (
        TASK_REGISTRY,
        collect_metrics,
        common_model_ids,
        default_charts_dir,
        folder_labels,
        payload_task,
        render_charts,
    )
except ModuleNotFoundError:
    from compare_metrics import (
        TASK_REGISTRY,
        collect_metrics,
        common_model_ids,
        default_charts_dir,
        folder_labels,
        payload_task,
        render_charts,
    )


def _detect_task_from_folders(folders: list[Path]) -> str:
    """Detect the benchmark task from result payloads, falling back to text-generation."""

    detected_tasks: set[str] = set()
    for folder in folders:
        for path in sorted(folder.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as file:
                    payload: Any = json.load(file)
            except (OSError, JSONDecodeError) as exc:
                print(f"Warning: failed to parse {path}: {exc}")
                continue
            if not isinstance(payload, Mapping):
                continue
            task = payload_task(payload)
            if task is None:
                continue
            if task not in TASK_REGISTRY:
                raise SystemExit(f"Unsupported task '{task}' found in {path}.")
            detected_tasks.add(task)

    if not detected_tasks:
        return "text-generation"
    if len(detected_tasks) > 1:
        tasks = ", ".join(sorted(detected_tasks))
        raise SystemExit(f"Multiple task values found ({tasks}); please pass --task explicitly.")
    task = next(iter(detected_tasks))
    print(f"Auto-detected task: {task}")
    return task


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare N benchmark result folders and generate model-wise bar charts."
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="benchmark result folders (relative/absolute). Pass 2 or more.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=("directory to save PNG charts (default: benchmark/transformers/results/charts/<folder1_folder2_...>)"),
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_REGISTRY.keys()),
        default=None,
        help=(
            "which benchmark payload to compare "
            "(default: auto-detect from task, fallback: text-generation)"
        ),
    )
    parser.add_argument(
        "--strip-owner",
        action="store_true",
        help="compare models by repository name only, ignoring leading Hugging Face owner ids",
    )
    args = parser.parse_args()

    folders = [Path(folder).expanduser().resolve() for folder in args.folders]
    if len(folders) < 2:
        raise SystemExit("Please provide at least 2 folders.")
    for folder in folders:
        if not folder.is_dir():
            raise SystemExit(f"Not a directory: {folder}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_charts_dir(Path(__file__).resolve().parent, folders)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    task = args.task or _detect_task_from_folders(folders)
    metric_cls = TASK_REGISTRY[task]
    metrics_by_folder = [collect_metrics(folder, metric_cls, strip_owner=args.strip_owner) for folder in folders]
    labels = folder_labels(folders)
    models = common_model_ids(metrics_by_folder)
    for label, folder, metrics in zip(labels, folders, metrics_by_folder):
        print(f"Folder {label}: {folder} -> {len(metrics)} models")
    print(f"Common models across all folders: {len(models)}")
    for model in models:
        print(f" - {model}")
    if not models:
        for label, metrics in zip(labels, metrics_by_folder):
            sample = sorted(metrics.keys())[:10]
            print(f"[debug] {label} normalized keys (up to 10): {sample}")
        raise SystemExit("No common model_id found across all input folders.")

    render_charts(
        metric_cls=metric_cls,
        models=models,
        labels=labels,
        metrics_by_folder=metrics_by_folder,
        output_dir=output_dir,
    )

    print(f"Saved charts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
