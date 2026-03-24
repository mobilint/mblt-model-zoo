from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

from mblt_model_zoo.hf_transformers.utils import list_models
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    add_device_tracking_args as _add_device_tracking_args,
    add_pipeline_device_args as _add_pipeline_device_args,
    build_device_tracker as _build_device_tracker,
    extract_device_metric as _extract_device_metric,
    parse_positive_int as _parse_positive_int,
    parse_positive_int_optional as _parse_positive_int_optional,
    print_device_status as _print_device_status,
    stop_tracker_safe as _stop_tracker_safe,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import VLMTPSMeasurer


def _safe_filename(text: str) -> str:
    return text.replace("/", "__")


def _parse_int_csv(raw: str) -> list[int]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    values = [int(x) for x in parts]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return sorted(set(values))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _is_cuda_device(device: str | None) -> bool:
    return isinstance(device, str) and device.strip().lower().startswith("cuda")


def _clear_cuda_memory(device: str | None) -> None:
    if not _is_cuda_device(device):
        return
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _release_pipeline(pipeline_obj: Any, device: str | None) -> None:
    if pipeline_obj is not None:
        try:
            model_obj = getattr(pipeline_obj, "model", None)
            if model_obj is not None and hasattr(model_obj, "dispose"):
                model_obj.dispose()
            elif hasattr(pipeline_obj, "dispose"):
                pipeline_obj.dispose()
        except Exception:
            pass
        del pipeline_obj
    gc.collect()
    if _is_cuda_device(device):
        _clear_cuda_memory(device)


def _build_pipeline(args: argparse.Namespace, model_id: str, revision: str | None):
    kwargs: dict[str, Any] = {
        "task": "image-text-to-text",
        "model": model_id,
        "trust_remote_code": args.trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    if args.tokenizer:
        kwargs["tokenizer"] = args.tokenizer
    if args.device is not None:
        kwargs["device"] = args.device
    if args.device_map:
        kwargs["device_map"] = args.device_map
    model_kwargs: dict[str, Any] = {}
    if args.core_mode:
        model_kwargs["core_mode"] = args.core_mode
        if args.core_mode == "single":
            model_kwargs["target_cores"] = ["0:0"]
        elif args.core_mode == "global4":
            model_kwargs["target_clusters"] = [0]
        elif args.core_mode == "global8":
            model_kwargs["target_clusters"] = [0, 1]
    if args.mxq_path:
        model_kwargs["mxq_path"] = args.mxq_path
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    if args.dtype:
        try:
            kwargs["dtype"] = args.dtype
            return hf_pipeline(**kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = args.dtype
    return hf_pipeline(**kwargs)


def _iter_targets(model_ids: list[str], revision: str | None, all_revisions: bool) -> list[tuple[str, str | None, str, str]]:
    out: list[tuple[str, str | None, str, str]] = []
    if not all_revisions:
        for m in model_ids:
            out.append((m, revision, m, _safe_filename(m)))
        return out
    for m in model_ids:
        out.append((m, "W8", f"{m}-W8", f"{_safe_filename(m)}-W8"))
        out.append((m, "W4V8", f"{m}-W4V8", f"{_safe_filename(m)}-W4V8"))
    return out


def _run_model(args: argparse.Namespace, label: str, pipeline: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    measurer = VLMTPSMeasurer(pipeline)
    tracker = _build_device_tracker(args, pipeline)
    _print_device_status(args, tracker)

    csv_rows: list[dict[str, Any]] = []
    vision_results: list[dict[str, Any]] = []

    for resolution in args.image_resolutions:
        for _ in range(args.warmup):
            measurer.measure_vision(image_resolution=resolution, repeat=1, prompt=args.prompt, show_progress=False)
        runs = []
        for _ in range(args.repeat):
            if tracker is not None:
                tracker.start()
            try:
                run = measurer.measure_vision(image_resolution=resolution, repeat=1, prompt=args.prompt, show_progress=False)[0]
            finally:
                _stop_tracker_safe(tracker)
            runs.append(run)
        encode_ms = [float(lat * 1000.0) for lat, _ in runs]
        fps = [float(v) for _, v in runs]
        vision_results.append({"image_resolution": resolution, "runs": runs, "summary": {"vision_encode_ms_mean": _mean(encode_ms), "vision_fps_mean": _mean(fps)}})
        for idx, (lat, vfps) in enumerate(runs, start=1):
            csv_rows.append({"type": "vision", "image_resolution": resolution, "repeat_index": idx, "vision_encode_ms": lat * 1000.0, "vision_fps": vfps})

    llm_resolution = args.llm_resolution if args.llm_resolution is not None else args.image_resolutions[0]
    for _ in range(args.warmup):
        measurer.measure_llm(image_resolution=llm_resolution, num_decode=args.decode, repeat=1, prompt=args.prompt, show_progress=False)

    llm_runs = []
    for _ in range(args.repeat):
        if tracker is not None:
            tracker.start()
        try:
            run = measurer.measure_llm(image_resolution=llm_resolution, num_decode=args.decode, repeat=1, prompt=args.prompt, show_progress=False)[0]
        finally:
            _stop_tracker_safe(tracker)
        if tracker is not None:
            metric = _extract_device_metric(tracker)
            run.avg_power_w = metric.get("avg_power_w")
        llm_runs.append(run)
    llm_prefill = [float(r.prefill_tps) for r in llm_runs]
    llm_decode = [float(r.decode_tps) for r in llm_runs]
    llm_ttft_ms = [float(r.prefill_latency * 1000.0) for r in llm_runs]
    for idx, r in enumerate(llm_runs, start=1):
        csv_rows.append({"type": "llm", "image_resolution": llm_resolution, "repeat_index": idx, "llm_prefill_tps": r.prefill_tps, "llm_decode_tps": r.decode_tps, "llm_ttft_ms": r.prefill_latency * 1000.0})

    payload = {
        "model": label,
        "task": "image-text-to-text",
        "benchmark": {
            "prompt": args.prompt,
            "decode": args.decode,
            "llm_reference_resolution": llm_resolution,
            "vision_results": vision_results,
            "llm_results": {"runs": [asdict(r) for r in llm_runs], "summary": {"llm_prefill_tps_mean": _mean(llm_prefill), "llm_decode_tps_mean": _mean(llm_decode), "llm_ttft_ms_mean": _mean(llm_ttft_ms)}},
        },
    }
    return payload, csv_rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rebuild_combined(results_dir: Path) -> None:
    llm_rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        bench = payload.get("benchmark", {})
        summ = bench.get("llm_results", {}).get("summary", {})
        llm_rows.append({"model": payload.get("model"), "llm_prefill_tps_mean": summ.get("llm_prefill_tps_mean"), "llm_decode_tps_mean": summ.get("llm_decode_tps_mean"), "llm_ttft_ms_mean": summ.get("llm_ttft_ms_mean")})
    _write_csv(results_dir / "combined_llm.csv", llm_rows)
    if not llm_rows:
        return
    models = [str(r["model"]) for r in llm_rows]
    decode = [float(r.get("llm_decode_tps_mean") or 0.0) for r in llm_rows]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(models) + 2)))
    y = range(len(models))
    ax.barh(list(y), decode)
    ax.set_yticks(list(y))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("tok/s")
    ax.set_title("LLM Decode TPS (mean)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(results_dir / "combined_llm_decode_tps_mean.png", dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark image-text-to-text models.")
    _add_pipeline_device_args(parser, device_default=None, trust_remote_code_default=True)
    _add_device_tracking_args(parser)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mxq-path", default=None)
    parser.add_argument("--core-mode", choices=["single", "multi", "global4", "global8"], default=None)
    parser.add_argument("--image-resolutions", type=_parse_int_csv, default=_parse_int_csv("224,384,512,768"))
    parser.add_argument("--decode", type=_parse_positive_int, default=128)
    parser.add_argument("--llm-resolution", type=_parse_positive_int_optional, default=None)
    parser.add_argument("--prompt", default="Describe the image in one sentence.")
    parser.add_argument("--warmup", type=_parse_positive_int, default=1)
    parser.add_argument("--repeat", type=_parse_positive_int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rebuild-charts", action="store_true")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    script_dir = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir).resolve() if args.results_dir else script_dir / "results" / "image_text_to_text"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_charts:
        _rebuild_combined(results_dir)
        return 0

    if args.model:
        model_ids = [str(args.model)]
    else:
        model_ids = list_models(tasks="image-text-to-text").get("image-text-to-text", [])
    if not model_ids:
        print("No image-text-to-text models found.")
        return 0

    for model_id, revision, label, base in tqdm(_iter_targets(model_ids, args.revision, args.all), desc="Benchmarking VLM models", unit="model"):
        if _is_cuda_device(args.device):
            _clear_cuda_memory(args.device)
        json_path = results_dir / f"{base}.json"
        csv_path = results_dir / f"{base}.csv"
        if args.skip_existing and json_path.is_file() and csv_path.is_file():
            print(f"Skipping {label} (results exist).")
            continue
        pipeline = None
        try:
            pipeline = _build_pipeline(args, model_id, revision)
            payload, rows = _run_model(args, label, pipeline)
            _write_json(json_path, payload)
            _write_csv(csv_path, rows)
            print(f"Saved: {json_path.name}, {csv_path.name}")
        except Exception as e:
            print(f"Skipping {label} (benchmark failed): {e}")
        finally:
            _release_pipeline(pipeline, args.device)

    _rebuild_combined(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
