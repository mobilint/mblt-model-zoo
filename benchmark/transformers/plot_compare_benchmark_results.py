import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from chart_utils import (
    ModelMetrics,
    common_models,
    default_charts_dir,
    folder_labels,
    load_model_metrics,
    plot_scalar_chart,
    plot_token_chart,
)


def _strip_group_id(model_id: str) -> str:
    # Compare by model id only (ignore leading group_id__ prefix).
    return model_id.split("__", 1)[1] if "__" in model_id else model_id


def _normalize_model_key(path: Path, loaded_model_id: str) -> str:
    # Prefer file name normalization so compare behavior is stable even if JSON "model" differs.
    stem = path.stem
    if "__" in stem:
        return _strip_group_id(stem)
    key = _strip_group_id(loaded_model_id)
    if "/" in key:
        key = key.split("/", 1)[1]
    return key


def _collect_compare_metrics(folder: Path) -> dict[str, ModelMetrics]:
    normalized: dict[str, ModelMetrics] = {}
    for path in sorted(folder.glob("*.json")):
        try:
            loaded = load_model_metrics(path)
        except Exception as e:
            print(f"Warning: failed to parse {path}: {e}")
            continue
        if loaded is None:
            continue
        model_id, metric = loaded
        norm_key = _normalize_model_key(path, model_id)
        if norm_key not in normalized:
            normalized[norm_key] = metric
    return normalized


@dataclass
class VLMCompareMetrics:
    llm_prefill_tps: Optional[float]
    llm_decode_tps: Optional[float]
    llm_ttft_ms: Optional[float]
    llm_decode_duration_ms: Optional[float]
    llm_total_ms: Optional[float]
    vision_encode_ms: Optional[float]
    vision_fps: Optional[float]
    vision_img_per_j: Optional[float]
    vision_j_per_img: Optional[float]
    llm_prefill_tok_per_j: Optional[float]
    llm_decode_tok_per_j: Optional[float]
    llm_prefill_j_per_tok: Optional[float]
    llm_decode_j_per_tok: Optional[float]
    avg_power_w: Optional[float]
    total_energy_j: Optional[float]
    avg_utilization_pct: Optional[float]
    p99_utilization_pct: Optional[float]
    avg_memory_used_mb: Optional[float]
    p99_memory_used_mb: Optional[float]
    avg_memory_used_pct: Optional[float]
    p99_memory_used_pct: Optional[float]


def _as_float(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None


def _load_vlm_model_metrics(path: Path) -> Optional[tuple[str, VLMCompareMetrics]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    model_id = payload.get("model")
    if not isinstance(model_id, str) or not model_id:
        return None
    benchmark = payload.get("benchmark", {})
    if not isinstance(benchmark, dict):
        return None
    llm_summary = benchmark.get("llm_results", {}).get("summary", {})
    vision_summary = benchmark.get("vision_summary", {})
    device = payload.get("device", {})
    if not isinstance(llm_summary, dict):
        llm_summary = {}
    if not isinstance(vision_summary, dict):
        vision_summary = {}
    if not isinstance(device, dict):
        device = {}
    return model_id, VLMCompareMetrics(
        llm_prefill_tps=_as_float(
            llm_summary.get("llm_prefill_tps", {}).get("mean")
            if isinstance(llm_summary.get("llm_prefill_tps"), dict)
            else None
        ),
        llm_decode_tps=_as_float(
            llm_summary.get("llm_decode_tps", {}).get("mean")
            if isinstance(llm_summary.get("llm_decode_tps"), dict)
            else None
        ),
        llm_ttft_ms=_as_float(
            llm_summary.get("llm_ttft_ms", {}).get("mean") if isinstance(llm_summary.get("llm_ttft_ms"), dict) else None
        ),
        llm_decode_duration_ms=_as_float(
            llm_summary.get("llm_decode_duration_ms", {}).get("mean")
            if isinstance(llm_summary.get("llm_decode_duration_ms"), dict)
            else None
        ),
        llm_total_ms=_as_float(
            llm_summary.get("llm_total_ms", {}).get("mean")
            if isinstance(llm_summary.get("llm_total_ms"), dict)
            else None
        ),
        vision_encode_ms=_as_float(
            vision_summary.get("vision_encode_ms", {}).get("mean")
            if isinstance(vision_summary.get("vision_encode_ms"), dict)
            else None
        ),
        vision_fps=_as_float(
            vision_summary.get("vision_fps", {}).get("mean")
            if isinstance(vision_summary.get("vision_fps"), dict)
            else None
        ),
        vision_img_per_j=_as_float(
            vision_summary.get("vision_img_per_j", {}).get("mean")
            if isinstance(vision_summary.get("vision_img_per_j"), dict)
            else None
        ),
        vision_j_per_img=_as_float(
            vision_summary.get("vision_j_per_img", {}).get("mean")
            if isinstance(vision_summary.get("vision_j_per_img"), dict)
            else None
        ),
        llm_prefill_tok_per_j=_as_float(
            llm_summary.get("prefill_tok_per_j", {}).get("mean")
            if isinstance(llm_summary.get("prefill_tok_per_j"), dict)
            else None
        ),
        llm_decode_tok_per_j=_as_float(
            llm_summary.get("decode_tok_per_j", {}).get("mean")
            if isinstance(llm_summary.get("decode_tok_per_j"), dict)
            else None
        ),
        llm_prefill_j_per_tok=_as_float(
            llm_summary.get("prefill_j_per_tok", {}).get("mean")
            if isinstance(llm_summary.get("prefill_j_per_tok"), dict)
            else None
        ),
        llm_decode_j_per_tok=_as_float(
            llm_summary.get("decode_j_per_tok", {}).get("mean")
            if isinstance(llm_summary.get("decode_j_per_tok"), dict)
            else None
        ),
        avg_power_w=_as_float(device.get("avg_power_w")),
        total_energy_j=_as_float(device.get("total_energy_j")),
        avg_utilization_pct=_as_float(device.get("avg_utilization_pct")),
        p99_utilization_pct=_as_float(device.get("p99_utilization_pct")),
        avg_memory_used_mb=_as_float(device.get("avg_memory_used_mb")),
        p99_memory_used_mb=_as_float(device.get("p99_memory_used_mb")),
        avg_memory_used_pct=_as_float(device.get("avg_memory_used_pct")),
        p99_memory_used_pct=_as_float(device.get("p99_memory_used_pct")),
    )


def _collect_compare_vlm_metrics(folder: Path) -> dict[str, VLMCompareMetrics]:
    normalized: dict[str, VLMCompareMetrics] = {}
    for path in sorted(folder.glob("*.json")):
        try:
            loaded = _load_vlm_model_metrics(path)
        except Exception as e:
            print(f"Warning: failed to parse {path}: {e}")
            continue
        if loaded is None:
            continue
        model_id, metric = loaded
        norm_key = _normalize_model_key(path, model_id)
        if norm_key not in normalized:
            normalized[norm_key] = metric
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
        help=("directory to save PNG charts (default: benchmark/transformers/results/charts/<folder1_folder2_...>)"),
    )
    parser.add_argument(
        "--task",
        choices=["text-generation", "image-text-to-text"],
        default="text-generation",
        help="which benchmark payload to compare (default: text-generation)",
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

    if args.task == "text-generation":
        metrics_by_folder = [_collect_compare_metrics(folder) for folder in folders]
    else:
        metrics_by_folder = [_collect_compare_vlm_metrics(folder) for folder in folders]
    labels = folder_labels(folders)
    models = common_models(metrics_by_folder)
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

    if args.task == "text-generation":
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
    else:
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.llm_prefill_tps,
            title="LLM Prefill TPS",
            x_label="TPS (tokens/sec)",
            output_path=output_dir / "llm_prefill_tps.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.llm_decode_tps,
            title="LLM Decode TPS",
            x_label="TPS (tokens/sec)",
            output_path=output_dir / "llm_decode_tps.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.llm_ttft_ms,
            title="LLM TTFT",
            x_label="Latency (ms)",
            output_path=output_dir / "llm_ttft_ms.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.llm_decode_duration_ms,
            title="LLM Decode Duration",
            x_label="Duration (ms)",
            output_path=output_dir / "llm_decode_duration_ms.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.vision_encode_ms,
            title="Vision Encode Latency",
            x_label="Latency (ms)",
            output_path=output_dir / "vision_encode_ms.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.vision_fps,
            title="Vision FPS",
            x_label="FPS",
            output_path=output_dir / "vision_fps.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.vision_img_per_j,
            title="Vision Img/J",
            x_label="img/J",
            output_path=output_dir / "vision_img_per_j.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.vision_j_per_img,
            title="Vision J/Img",
            x_label="J/img",
            output_path=output_dir / "vision_j_per_img.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.llm_prefill_tok_per_j,
            title="LLM Prefill Tokens/J",
            x_label="tok/J",
            output_path=output_dir / "llm_prefill_tokens_per_j.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.llm_decode_tok_per_j,
            title="LLM Decode Tokens/J",
            x_label="tok/J",
            output_path=output_dir / "llm_decode_tokens_per_j.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.llm_prefill_j_per_tok,
            title="LLM Prefill J/Token",
            x_label="J/tok",
            output_path=output_dir / "llm_prefill_j_per_token.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.llm_decode_j_per_tok,
            title="LLM Decode J/Token",
            x_label="J/tok",
            output_path=output_dir / "llm_decode_j_per_token.png",
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
