import argparse
from pathlib import Path

from chart_utils import (
    collect_folder_metrics,
    common_models,
    default_charts_dir,
    folder_labels,
    plot_scalar_chart,
    plot_token_chart,
)


def _strip_group_id(model_id: str) -> str:
    # Compare by model id only (ignore leading group_id__ prefix).
    return model_id.split("__", 1)[1] if "__" in model_id else model_id


def _normalize_folder_metrics(metrics: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in metrics.items():
        norm_key = _strip_group_id(str(key))
        if norm_key not in normalized:
            normalized[norm_key] = value
    return normalized


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
            "directory to save PNG charts "
            "(default: benchmark/transformers/results/charts/<folder1_folder2_...>)"
        ),
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

    metrics_by_folder = [
        _normalize_folder_metrics(collect_folder_metrics(folder)) for folder in folders
    ]
    labels = folder_labels(folders)
    models = common_models(metrics_by_folder)
    for label, folder, metrics in zip(labels, folders, metrics_by_folder):
        print(f"Folder {label}: {folder} -> {len(metrics)} models")
    print(f"Common models across all folders: {len(models)}")
    for model in models:
        print(f" - {model}")
    if not models:
        raise SystemExit("No common model_id found across all input folders.")

    plot_token_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        token_selector=lambda m: m.prefill_tps,
        title="Prefill TPS",
        x_label="TPS (tokens/sec)",
        output_path=output_dir / "prefill_tps.png",
    )
    plot_token_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        token_selector=lambda m: m.decode_tps,
        title="Decode TPS",
        x_label="TPS (tokens/sec)",
        output_path=output_dir / "decode_tps.png",
    )
    plot_token_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        token_selector=lambda m: m.prefill_latency_ms,
        title="Prefill Latency",
        x_label="Latency (ms)",
        output_path=output_dir / "prefill_latency_ms.png",
    )
    plot_token_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        token_selector=lambda m: m.decode_duration_ms,
        title="Decode Duration",
        x_label="Duration (ms)",
        output_path=output_dir / "decode_duration_ms.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.avg_power_w,
        title="Average Power",
        x_label="Power (W)",
        output_path=output_dir / "avg_power_w.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.total_energy_j,
        title="Total Energy",
        x_label="Energy (J)",
        output_path=output_dir / "total_energy_j.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.prefill_tokens_per_j,
        title="Prefill Tokens/J",
        x_label="Tokens/J",
        output_path=output_dir / "prefill_tokens_per_j.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.decode_tokens_per_j,
        title="Decode Tokens/J",
        x_label="Tokens/J",
        output_path=output_dir / "decode_tokens_per_j.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.prefill_j_per_token,
        title="Prefill J/Token",
        x_label="J/Token",
        output_path=output_dir / "prefill_j_per_token.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.decode_j_per_token,
        title="Decode J/Token",
        x_label="J/Token",
        output_path=output_dir / "decode_j_per_token.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.avg_utilization_pct,
        title="Average Utilization",
        x_label="Utilization (%)",
        output_path=output_dir / "avg_utilization_pct.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.p99_utilization_pct,
        title="P99 Utilization",
        x_label="Utilization (%)",
        output_path=output_dir / "p99_utilization_pct.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.avg_memory_used_mb,
        title="Average Memory Used",
        x_label="Memory (MB)",
        output_path=output_dir / "avg_memory_used_mb.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.p99_memory_used_mb,
        title="P99 Memory Used",
        x_label="Memory (MB)",
        output_path=output_dir / "p99_memory_used_mb.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.avg_memory_used_pct,
        title="Average Memory Used (%)",
        x_label="Memory Usage (%)",
        output_path=output_dir / "avg_memory_used_pct.png",
    )
    plot_scalar_chart(
        models=models,
        folder_labels=labels,
        metrics_by_folder=metrics_by_folder,
        scalar_selector=lambda m: m.p99_memory_used_pct,
        title="P99 Memory Used (%)",
        x_label="Memory Usage (%)",
        output_path=output_dir / "p99_memory_used_pct.png",
    )

    print(f"Saved charts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
