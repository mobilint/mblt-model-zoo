import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from benchmark.common.summary_utils import (
        HOST_PC_INFO_FILENAME,
        existing_png_paths,
        write_summary_markdown,
    )
    from benchmark.transformers.compare_metrics import (
        TASK_REGISTRY,
        build_compare_plot_tables,
        collect_metrics,
        common_model_ids,
        default_charts_dir,
        folder_labels,
        payload_task,
        render_charts,
        write_compare_markdown,
    )
except ModuleNotFoundError:
    from compare_metrics import (
        TASK_REGISTRY,
        build_compare_plot_tables,
        collect_metrics,
        common_model_ids,
        default_charts_dir,
        folder_labels,
        payload_task,
        render_charts,
        write_compare_markdown,
    )

    from benchmark.common.summary_utils import (
        HOST_PC_INFO_FILENAME,
        existing_png_paths,
        write_summary_markdown,
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


def _write_compare_host_info_json(output_dir: Path, labels: list[str], folders: list[Path]) -> Path:
    """Write source-labeled host info metadata collected from input folders."""

    payload: dict[str, Any] = {}
    for label, folder in zip(labels, folders):
        host_info_path = folder / HOST_PC_INFO_FILENAME
        entry: dict[str, Any] = {"source": host_info_path.as_posix()}
        if not host_info_path.is_file():
            entry.update({"status": "missing", "message": "host_pc_info.json was not found in the input folder."})
            payload[label] = entry
            continue
        try:
            with host_info_path.open("r", encoding="utf-8") as f:
                entry.update({"status": "ok", "payload": json.load(f)})
        except (OSError, json.JSONDecodeError) as exc:
            entry.update({"status": "error", "message": str(exc)})
        payload[label] = entry

    output_path = output_dir / HOST_PC_INFO_FILENAME
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved source Host PC Info: {output_path.name}")
    return output_path


def _model_key_without_owner(model_key: str) -> str:
    """Return a normalized model key without the leading Hugging Face owner id."""

    return model_key.rsplit("/", 1)[1] if "/" in model_key else model_key


def _common_model_keys_without_owner(metrics_by_folder: list[Mapping[str, Any]]) -> list[str]:
    """Return common normalized model keys after stripping leading owner ids."""

    if not metrics_by_folder:
        return []
    model_sets = [
        {_model_key_without_owner(model_key) for model_key in metrics}
        for metrics in metrics_by_folder
    ]
    if not model_sets:
        return []
    return sorted(set.intersection(*model_sets))


def _print_strip_owner_hint(metrics_by_folder: list[Mapping[str, Any]], *, limit: int = 10) -> None:
    """Print a hint when owner-stripped model keys would have common entries."""

    owner_stripped_models = _common_model_keys_without_owner(metrics_by_folder)
    if not owner_stripped_models:
        return

    print(
        "Hint: no common model_id was found, but common repository names were found after stripping "
        "leading Hugging Face owner ids. Did you mean to pass --strip-owner?"
    )
    print(f"[debug] --strip-owner common keys (up to {limit}): {owner_stripped_models[:limit]}")


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
        help=(
            "directory to save comparison outputs "
            "(default: benchmark/transformers/results/comparison/<folder1_folder2_...>)"
        ),
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
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="save PNG charts only and skip combined.md, host_pc_info.json, and summary.md generation",
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
        if not args.strip_owner:
            _print_strip_owner_hint(metrics_by_folder)
        raise SystemExit("No common model_id found across all input folders.")

    render_charts(
        metric_cls=metric_cls,
        models=models,
        labels=labels,
        metrics_by_folder=metrics_by_folder,
        output_dir=output_dir,
    )

    if not args.no_summary:
        combined_md = output_dir / "combined.md"
        write_compare_markdown(
            combined_md,
            metric_cls=metric_cls,
            models=models,
            labels=labels,
            metrics_by_folder=metrics_by_folder,
        )
        compare_host_info_path = _write_compare_host_info_json(output_dir, labels, folders)
        write_summary_markdown(
            output_dir / "summary.md",
            title=f"{task} Benchmark Comparison Summary",
            host_info_path=compare_host_info_path,
            table_markdown_path=combined_md,
            plot_paths=existing_png_paths(output_dir),
            plot_tables=build_compare_plot_tables(
                metric_cls=metric_cls,
                models=models,
                labels=labels,
                metrics_by_folder=metrics_by_folder,
            ),
            host_info_paths={label: folder / HOST_PC_INFO_FILENAME for label, folder in zip(labels, folders)},
        )

    print(f"Saved comparison outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
