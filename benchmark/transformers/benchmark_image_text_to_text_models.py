from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

from benchmark.common.argparse_utils import parse_int_csv as _parse_int_csv
from benchmark.common.argparse_utils import parse_range_arg as _parse_range_arg
from benchmark.common.chart_utils import plot_simple_barh
from benchmark.common.io_utils import safe_filename as _safe_filename_common
from benchmark.common.io_utils import write_csv as _write_csv
from benchmark.common.io_utils import write_json as _write_json
from benchmark.common.math_utils import safe_div as _safe_div
from benchmark.common.runtime_utils import clear_cuda_memory as _clear_cuda_memory
from benchmark.common.runtime_utils import is_cuda_device as _is_cuda_device
from benchmark.common.runtime_utils import release_pipeline as _release_pipeline
from benchmark.common.summary_utils import HOST_PC_INFO_FILENAME as _HOST_PC_INFO_FILENAME
from benchmark.common.summary_utils import collect_host_pc_info as _collect_host_pc_info
from benchmark.common.summary_utils import existing_png_paths as _existing_png_paths
from benchmark.common.summary_utils import write_summary_markdown as _write_summary_markdown
from mblt_model_zoo.hf_transformers.utils import list_models
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    CORE_MODE_CHOICES as _CORE_MODE_CHOICES_COMMON,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    add_device_tracking_args as _add_device_tracking_args,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    add_pipeline_device_args as _add_pipeline_device_args,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    append_core_mode_suffix as _append_core_mode_suffix_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    apply_core_mode_model_kwargs as _apply_core_mode_model_kwargs_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    build_device_tracker as _build_device_tracker,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    extract_device_metric as _extract_device_metric,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    extract_device_time_series as _extract_device_time_series,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    iter_core_modes as _iter_core_modes_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    parse_positive_int as _parse_positive_int,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    parse_positive_int_optional as _parse_positive_int_optional,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    print_device_status as _print_device_status,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    resolve_default_device as _resolve_default_device_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    resolve_default_device_backend as _resolve_default_device_backend_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    stop_tracker_safe as _stop_tracker_safe,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import VLMTPSMeasurer, npu_latency_pct

try:
    from benchmark_text_generation_models import (
        _add_batch_selection_args,
        _cuda_memory_info,
        _estimate_model_weight_bytes,
        _filter_text_targets_by_batch_mode,
        _format_gib,
        _is_cuda_oom_error,
        _iter_targets_from_mxq_dir,
        _read_raw_config,
        _resolve_original_model_ids,
        _revision_exists,
        _should_precheck_cuda,
    )
except Exception:
    from .benchmark_text_generation_models import (
        _add_batch_selection_args,
        _cuda_memory_info,
        _estimate_model_weight_bytes,
        _filter_text_targets_by_batch_mode,
        _format_gib,
        _is_cuda_oom_error,
        _iter_targets_from_mxq_dir,
        _read_raw_config,
        _resolve_original_model_ids,
        _revision_exists,
        _should_precheck_cuda,
    )


@dataclass(frozen=True)
class VLMBenchmarkTarget:
    """Resolved image-text-to-text benchmark target with batch metadata."""

    model_id: str
    revision: str | None
    label: str
    base: str
    mxq_path: str | None
    max_batch_size: int


def _safe_filename(text: str) -> str:
    return _safe_filename_common(text, replace_slash_only=True)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summary(values: list[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": _mean(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "p50": _percentile(vals, 0.50),
        "p95": _percentile(vals, 0.95),
        "p99": _percentile(vals, 0.99),
    }


def _build_pipeline(
    args: argparse.Namespace,
    model_id: str,
    revision: str | None,
    mxq_path: str | None,
    core_mode: str | None,
):
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
    model_kwargs = _apply_vlm_core_mode_model_kwargs(model_kwargs, core_mode)
    if mxq_path:
        model_kwargs["mxq_path"] = mxq_path
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


def _apply_vlm_core_mode_model_kwargs(model_kwargs: dict[str, Any], core_mode: str | None) -> dict[str, Any]:
    """Apply shared VLM NPU core-mode kwargs to both vision and text sub-configs.

    Image-text-to-text Mobilint models use composite configs with separate ``vision_config`` and
    ``text_config`` NPU backends. Passing text-generation style top-level kwargs such as
    ``core_mode`` leaves them unused by the config loader, and Transformers forwards them to the
    model constructor where upstream VLM classes can reject them. This helper maps the benchmark's
    shared ``--core-mode`` option onto the VLM-specific ``vision_*`` and ``text_*`` config fields.

    Args:
        model_kwargs: Existing model kwargs to update.
        core_mode: Core mode requested by the benchmark CLI.

    Returns:
        The updated model kwargs.
    """
    expanded: dict[str, Any] = {}
    _apply_core_mode_model_kwargs_common(expanded, core_mode)

    for prefix in ("vision", "text"):
        for key, value in expanded.items():
            model_kwargs[f"{prefix}_{key}"] = value
    return model_kwargs


def _collect_vlm_config_mxq_paths(config: dict[str, Any]) -> list[str]:
    """Collect MXQ paths referenced by a VLM config and its sub-configs."""
    paths: list[str] = []

    def _visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key == "mxq_path" and isinstance(item, str) and item:
                paths.append(item)
            elif key.endswith("_config") and isinstance(item, dict):
                _visit(item)

    _visit(config)
    return paths


def _list_repo_files(model_id: str, revision: str | None) -> list[str] | None:
    """Return repository files for a revision, or ``None`` when Hub access fails."""
    try:
        from huggingface_hub import HfApi

        return list(HfApi().list_repo_files(repo_id=model_id, revision=revision, repo_type="model"))
    except Exception:
        return None


def _vlm_revision_artifacts_available(
    model_id: str,
    revision: str | None,
    mxq_path: str | None,
) -> tuple[bool, str | None]:
    """Best-effort preflight check for VLM revision and referenced MXQ artifacts."""
    if mxq_path or not revision:
        return True, None

    revision_state = _revision_exists(model_id, revision)
    if revision_state is False:
        return False, f"revision {revision!r} was not found on Hugging Face Hub"

    config = _read_raw_config(model_id, revision)
    if config is None:
        return True, None

    mxq_paths = _collect_vlm_config_mxq_paths(config)
    if not mxq_paths:
        return True, None

    revision_files = _list_repo_files(model_id, revision)
    main_files = _list_repo_files(model_id, None)
    if revision_files is None and main_files is None:
        return True, None

    file_sets = [set(files) for files in (revision_files, main_files) if files is not None]
    missing = [
        path
        for path in mxq_paths
        if not any(path in files or os.path.basename(path) in files for files in file_sets)
    ]
    if missing:
        return False, f"referenced MXQ artifact(s) not found: {', '.join(missing)}"
    return True, None


def _iter_targets(
    model_ids: list[str], revision: str | None, all_revisions: bool
) -> list[tuple[str, str | None, str, str]]:
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

    all_vision_encode_ms: list[float] = []
    all_vision_fps: list[float] = []
    all_vision_avg_power_w: list[float] = []
    all_vision_p99_power_w: list[float] = []
    all_vision_avg_utilization_pct: list[float] = []
    all_vision_p99_utilization_pct: list[float] = []
    all_vision_avg_temperature_c: list[float] = []
    all_vision_p99_temperature_c: list[float] = []
    all_vision_avg_memory_used_mb: list[float] = []
    all_vision_p99_memory_used_mb: list[float] = []
    all_vision_avg_memory_used_pct: list[float] = []
    all_vision_p99_memory_used_pct: list[float] = []
    all_vision_total_energy_j: list[float] = []
    all_vision_img_per_j: list[float] = []
    all_vision_j_per_img: list[float] = []

    for resolution in args.image_resolutions:
        for _ in tqdm(
            range(args.warmup),
            desc=f"{label} vision@{resolution} warmup",
            leave=False,
        ):
            measurer.measure_vision(
                image_resolution=resolution,
                repeat=1,
                prompt=args.prompt,
                batch_size=args.batch_size,
                show_progress=False,
            )
        runs = []
        power_vals: list[float] = []
        p99_power_vals: list[float] = []
        util_vals: list[float] = []
        p99_util_vals: list[float] = []
        temp_vals: list[float] = []
        p99_temp_vals: list[float] = []
        mem_mb_vals: list[float] = []
        p99_mem_mb_vals: list[float] = []
        mem_pct_vals: list[float] = []
        p99_mem_pct_vals: list[float] = []
        energy_vals: list[float] = []
        img_per_j_vals: list[float] = []
        j_per_img_vals: list[float] = []
        device_time_series_runs: list[dict[str, list[dict[str, float]]]] = []
        for _ in tqdm(
            range(args.repeat),
            desc=f"{label} vision@{resolution} runs",
            leave=False,
        ):
            if tracker is not None:
                tracker.start()
            try:
                run = measurer.measure_vision(
                    image_resolution=resolution,
                    repeat=1,
                    prompt=args.prompt,
                    batch_size=args.batch_size,
                    show_progress=False,
                )[0]
            finally:
                _stop_tracker_safe(tracker)
            runs.append(run)
            if tracker is not None:
                metric = _extract_device_metric(tracker)
                device_time_series_runs.append(_extract_device_time_series(tracker))
                avg_power = metric.get("avg_power_w")
                p99_power = metric.get("p99_power_w")
                avg_util = metric.get("avg_utilization_pct")
                p99_util = metric.get("p99_utilization_pct")
                avg_temp = metric.get("avg_temperature_c")
                p99_temp = metric.get("p99_temperature_c")
                avg_mem = metric.get("avg_memory_used_mb")
                p99_mem = metric.get("p99_memory_used_mb")
                avg_mem_pct = metric.get("avg_memory_used_pct")
                p99_mem_pct = metric.get("p99_memory_used_pct")
                if avg_power is not None:
                    power_vals.append(float(avg_power))
                    energy = float(avg_power) * float(run[0])
                    energy_vals.append(energy)
                    j_per_img_vals.append(energy)
                    tpj = _safe_div(1.0, energy)
                    if tpj is not None:
                        img_per_j_vals.append(tpj)
                if p99_power is not None:
                    p99_power_vals.append(float(p99_power))
                if avg_util is not None:
                    util_vals.append(float(avg_util))
                if p99_util is not None:
                    p99_util_vals.append(float(p99_util))
                if avg_temp is not None:
                    temp_vals.append(float(avg_temp))
                if p99_temp is not None:
                    p99_temp_vals.append(float(p99_temp))
                if avg_mem is not None:
                    mem_mb_vals.append(float(avg_mem))
                if p99_mem is not None:
                    p99_mem_mb_vals.append(float(p99_mem))
                if avg_mem_pct is not None:
                    mem_pct_vals.append(float(avg_mem_pct))
                if p99_mem_pct is not None:
                    p99_mem_pct_vals.append(float(p99_mem_pct))
        encode_ms = [float(lat * 1000.0) for lat, _ in runs]
        fps = [float(v) for _, v in runs]
        all_vision_encode_ms.extend(encode_ms)
        all_vision_fps.extend(fps)
        all_vision_avg_power_w.extend(power_vals)
        all_vision_p99_power_w.extend(p99_power_vals)
        all_vision_avg_utilization_pct.extend(util_vals)
        all_vision_p99_utilization_pct.extend(p99_util_vals)
        all_vision_avg_temperature_c.extend(temp_vals)
        all_vision_p99_temperature_c.extend(p99_temp_vals)
        all_vision_avg_memory_used_mb.extend(mem_mb_vals)
        all_vision_p99_memory_used_mb.extend(p99_mem_mb_vals)
        all_vision_avg_memory_used_pct.extend(mem_pct_vals)
        all_vision_p99_memory_used_pct.extend(p99_mem_pct_vals)
        all_vision_total_energy_j.extend(energy_vals)
        all_vision_img_per_j.extend(img_per_j_vals)
        all_vision_j_per_img.extend(j_per_img_vals)
        vision_results.append(
            {
                "image_resolution": resolution,
                "runs": runs,
                "summary": {
                    "vision_encode_ms": _summary(encode_ms),
                    "vision_fps": _summary(fps),
                    "avg_power_w": _summary(power_vals),
                    "p99_power_w": _summary(p99_power_vals),
                    "avg_utilization_pct": _summary(util_vals),
                    "p99_utilization_pct": _summary(p99_util_vals),
                    "avg_temperature_c": _summary(temp_vals),
                    "p99_temperature_c": _summary(p99_temp_vals),
                    "avg_memory_used_mb": _summary(mem_mb_vals),
                    "p99_memory_used_mb": _summary(p99_mem_mb_vals),
                    "avg_memory_used_pct": _summary(mem_pct_vals),
                    "p99_memory_used_pct": _summary(p99_mem_pct_vals),
                    "total_energy_j": _summary(energy_vals),
                    "vision_img_per_j": _summary(img_per_j_vals),
                    "vision_j_per_img": _summary(j_per_img_vals),
                },
                "device_time_series_runs": device_time_series_runs,
            }
        )
        for idx, (lat, vfps) in enumerate(runs, start=1):
            csv_rows.append(
                {
                    "type": "vision",
                    "image_resolution": resolution,
                    "repeat_index": idx,
                    "vision_encode_ms": lat * 1000.0,
                    "vision_fps": vfps,
                    "llm_prefill_tps": None,
                    "llm_decode_tps": None,
                    "llm_ttft_ms": None,
                    "llm_prefill_npu_latency_pct": None,
                    "llm_decode_npu_latency_pct": None,
                    "avg_power_w": power_vals[idx - 1] if idx - 1 < len(power_vals) else None,
                    "p99_power_w": p99_power_vals[idx - 1] if idx - 1 < len(p99_power_vals) else None,
                    "avg_utilization_pct": util_vals[idx - 1] if idx - 1 < len(util_vals) else None,
                    "p99_utilization_pct": p99_util_vals[idx - 1] if idx - 1 < len(p99_util_vals) else None,
                    "avg_temperature_c": temp_vals[idx - 1] if idx - 1 < len(temp_vals) else None,
                    "p99_temperature_c": p99_temp_vals[idx - 1] if idx - 1 < len(p99_temp_vals) else None,
                    "avg_memory_used_mb": mem_mb_vals[idx - 1] if idx - 1 < len(mem_mb_vals) else None,
                    "p99_memory_used_mb": p99_mem_mb_vals[idx - 1] if idx - 1 < len(p99_mem_mb_vals) else None,
                    "avg_memory_used_pct": mem_pct_vals[idx - 1] if idx - 1 < len(mem_pct_vals) else None,
                    "p99_memory_used_pct": p99_mem_pct_vals[idx - 1] if idx - 1 < len(p99_mem_pct_vals) else None,
                    "total_energy_j": energy_vals[idx - 1] if idx - 1 < len(energy_vals) else None,
                    "prefill_tok_per_j": None,
                    "decode_tok_per_j": None,
                    "prefill_j_per_tok": None,
                    "decode_j_per_tok": None,
                    "vision_img_per_j": img_per_j_vals[idx - 1] if idx - 1 < len(img_per_j_vals) else None,
                    "vision_j_per_img": j_per_img_vals[idx - 1] if idx - 1 < len(j_per_img_vals) else None,
                }
            )

    llm_resolution = args.llm_resolution if args.llm_resolution is not None else args.image_resolutions[0]
    resolved_prefill_chunk_size = None if args.original_models and not args.mxq_dir else args.prefill_chunk_size
    for warmup_idx in tqdm(
        range(args.warmup),
        desc=f"{label} llm@{llm_resolution} warmup",
        leave=False,
    ):
        measurer.measure_llm_full(
            image_resolution=llm_resolution,
            prompt=args.prompt,
            prefill_range=args.prefill_range,
            cache_lengths=args.cache_lengths,
            decode_window=args.decode_window,
            prefill_chunk_size=resolved_prefill_chunk_size,
            batch_size=args.batch_size,
            show_progress=True,
            progress_prefix=f"{label} llm@{llm_resolution} warmup {warmup_idx + 1}/{args.warmup}",
        )

    llm_runs = []
    llm_avg_power_w: list[float] = []
    llm_p99_power_w: list[float] = []
    llm_avg_utilization_pct: list[float] = []
    llm_p99_utilization_pct: list[float] = []
    llm_avg_temperature_c: list[float] = []
    llm_p99_temperature_c: list[float] = []
    llm_avg_memory_used_mb: list[float] = []
    llm_p99_memory_used_mb: list[float] = []
    llm_avg_memory_used_pct: list[float] = []
    llm_p99_memory_used_pct: list[float] = []
    llm_total_energy_j: list[float] = []
    llm_prefill_tok_per_j: list[float] = []
    llm_decode_tok_per_j: list[float] = []
    llm_prefill_j_per_tok: list[float] = []
    llm_decode_j_per_tok: list[float] = []
    llm_device_time_series_runs: list[dict[str, list[dict[str, float]]]] = []
    for repeat_idx in tqdm(
        range(args.repeat),
        desc=f"{label} llm@{llm_resolution} runs",
        leave=False,
    ):
        if tracker is not None:
            tracker.start()
        try:
            run = measurer.measure_llm_full(
                image_resolution=llm_resolution,
                prompt=args.prompt,
                prefill_range=args.prefill_range,
                cache_lengths=args.cache_lengths,
                decode_window=args.decode_window,
                prefill_chunk_size=resolved_prefill_chunk_size,
                batch_size=args.batch_size,
                show_progress=True,
                progress_prefix=f"{label} llm@{llm_resolution} run {repeat_idx + 1}/{args.repeat}",
            )
        finally:
            _stop_tracker_safe(tracker)
        if tracker is not None:
            metric = _extract_device_metric(tracker)
            llm_device_time_series_runs.append(_extract_device_time_series(tracker))
            run.avg_power_w = metric.get("avg_power_w")
            run.p99_power_w = metric.get("p99_power_w")
            run.avg_utilization_pct = metric.get("avg_utilization_pct")
            run.p99_utilization_pct = metric.get("p99_utilization_pct")
            run.avg_temperature_c = metric.get("avg_temperature_c")
            run.p99_temperature_c = metric.get("p99_temperature_c")
            run.avg_memory_used_mb = metric.get("avg_memory_used_mb")
            run.p99_memory_used_mb = metric.get("p99_memory_used_mb")
            run.avg_memory_used_pct = metric.get("avg_memory_used_pct")
            run.p99_memory_used_pct = metric.get("p99_memory_used_pct")
            if run.avg_power_w is not None:
                total_time = float(run.prefill_phase_duration_s or 0.0) + float(run.decode_phase_duration_s or 0.0)
                e = float(run.avg_power_w) * total_time
                run.total_energy_j = e
                decode_last_tps = float(run.decode_sweep.tps_values[-1]) if run.decode_sweep.tps_values else 0.0
                prefill_last_tps = float(run.prefill_sweep.tps_values[-1]) if run.prefill_sweep.tps_values else 0.0
                t1 = _safe_div(prefill_last_tps, float(run.avg_power_w))
                t2 = _safe_div(decode_last_tps, float(run.avg_power_w))
                j1 = _safe_div(1.0, t1) if t1 not in (None, 0) else None
                j2 = _safe_div(1.0, t2) if t2 not in (None, 0) else None
                run.prefill_tokens_per_j = t1
                run.decode_tokens_per_j = t2
                run.prefill_j_per_token = j1
                run.decode_j_per_token = j2
        llm_runs.append(run)
    llm_prefill = [float(r.prefill_sweep.tps_values[-1]) for r in llm_runs if r.prefill_sweep.tps_values]
    llm_decode = [float(r.decode_sweep.tps_values[-1]) for r in llm_runs if r.decode_sweep.tps_values]
    llm_ttft_ms = [float(r.prefill_sweep.time_values[-1] * 1000.0) for r in llm_runs if r.prefill_sweep.time_values]
    llm_decode_ms = [float(r.decode_sweep.time_values[-1] * 1000.0) for r in llm_runs if r.decode_sweep.time_values]
    llm_prefill_npu_pct = [
        pct
        for r in llm_runs
        if r.prefill_sweep.avg_total_token_latency_values
        and r.prefill_sweep.avg_npu_token_latency_values
        and (
            pct := npu_latency_pct(
                r.prefill_sweep.avg_total_token_latency_values[-1],
                r.prefill_sweep.avg_npu_token_latency_values[-1],
            )
        )
        is not None
    ]
    llm_decode_npu_pct = [
        pct
        for r in llm_runs
        if r.decode_sweep.avg_total_token_latency_values
        and r.decode_sweep.avg_npu_token_latency_values
        and (
            pct := npu_latency_pct(
                r.decode_sweep.avg_total_token_latency_values[-1],
                r.decode_sweep.avg_npu_token_latency_values[-1],
            )
        )
        is not None
    ]
    llm_total_ms = [
        float((r.prefill_phase_duration_s or 0.0) + (r.decode_phase_duration_s or 0.0)) * 1000.0 for r in llm_runs
    ]
    llm_avg_power_w = [float(r.avg_power_w) for r in llm_runs if r.avg_power_w is not None]
    llm_p99_power_w = [float(r.p99_power_w) for r in llm_runs if r.p99_power_w is not None]
    llm_avg_utilization_pct = [float(r.avg_utilization_pct) for r in llm_runs if r.avg_utilization_pct is not None]
    llm_p99_utilization_pct = [float(r.p99_utilization_pct) for r in llm_runs if r.p99_utilization_pct is not None]
    llm_avg_temperature_c = [float(r.avg_temperature_c) for r in llm_runs if r.avg_temperature_c is not None]
    llm_p99_temperature_c = [float(r.p99_temperature_c) for r in llm_runs if r.p99_temperature_c is not None]
    llm_avg_memory_used_mb = [float(r.avg_memory_used_mb) for r in llm_runs if r.avg_memory_used_mb is not None]
    llm_p99_memory_used_mb = [float(r.p99_memory_used_mb) for r in llm_runs if r.p99_memory_used_mb is not None]
    llm_avg_memory_used_pct = [float(r.avg_memory_used_pct) for r in llm_runs if r.avg_memory_used_pct is not None]
    llm_p99_memory_used_pct = [float(r.p99_memory_used_pct) for r in llm_runs if r.p99_memory_used_pct is not None]
    llm_total_energy_j = [float(r.total_energy_j) for r in llm_runs if getattr(r, "total_energy_j", None) is not None]
    llm_prefill_tok_per_j = [
        float(r.prefill_tokens_per_j) for r in llm_runs if getattr(r, "prefill_tokens_per_j", None) is not None
    ]
    llm_decode_tok_per_j = [
        float(r.decode_tokens_per_j) for r in llm_runs if getattr(r, "decode_tokens_per_j", None) is not None
    ]
    llm_prefill_j_per_tok = [
        float(r.prefill_j_per_token) for r in llm_runs if getattr(r, "prefill_j_per_token", None) is not None
    ]
    llm_decode_j_per_tok = [
        float(r.decode_j_per_token) for r in llm_runs if getattr(r, "decode_j_per_token", None) is not None
    ]
    for idx, r in enumerate(llm_runs, start=1):
        llm_prefill_tps = float(r.prefill_sweep.tps_values[-1]) if r.prefill_sweep.tps_values else None
        llm_decode_tps = float(r.decode_sweep.tps_values[-1]) if r.decode_sweep.tps_values else None
        llm_ttft_ms = float(r.prefill_sweep.time_values[-1] * 1000.0) if r.prefill_sweep.time_values else None
        prefill_npu_pct = None
        if r.prefill_sweep.avg_total_token_latency_values and r.prefill_sweep.avg_npu_token_latency_values:
            prefill_npu_pct = npu_latency_pct(
                r.prefill_sweep.avg_total_token_latency_values[-1],
                r.prefill_sweep.avg_npu_token_latency_values[-1],
            )
        decode_npu_pct = None
        if r.decode_sweep.avg_total_token_latency_values and r.decode_sweep.avg_npu_token_latency_values:
            decode_npu_pct = npu_latency_pct(
                r.decode_sweep.avg_total_token_latency_values[-1],
                r.decode_sweep.avg_npu_token_latency_values[-1],
            )
        csv_rows.append(
            {
                "type": "llm",
                "image_resolution": llm_resolution,
                "repeat_index": idx,
                "vision_encode_ms": None,
                "vision_fps": None,
                "llm_prefill_tps": llm_prefill_tps,
                "llm_decode_tps": llm_decode_tps,
                "llm_ttft_ms": llm_ttft_ms,
                "llm_prefill_npu_latency_pct": prefill_npu_pct,
                "llm_decode_npu_latency_pct": decode_npu_pct,
                "avg_power_w": r.avg_power_w,
                "p99_power_w": r.p99_power_w,
                "avg_utilization_pct": r.avg_utilization_pct,
                "p99_utilization_pct": r.p99_utilization_pct,
                "avg_temperature_c": r.avg_temperature_c,
                "p99_temperature_c": r.p99_temperature_c,
                "avg_memory_used_mb": r.avg_memory_used_mb,
                "p99_memory_used_mb": r.p99_memory_used_mb,
                "avg_memory_used_pct": r.avg_memory_used_pct,
                "p99_memory_used_pct": r.p99_memory_used_pct,
                "total_energy_j": getattr(r, "total_energy_j", None),
                "prefill_tok_per_j": getattr(r, "prefill_tokens_per_j", None),
                "decode_tok_per_j": getattr(r, "decode_tokens_per_j", None),
                "prefill_j_per_tok": getattr(r, "prefill_j_per_token", None),
                "decode_j_per_tok": getattr(r, "decode_j_per_token", None),
                "vision_img_per_j": None,
            }
        )

    payload = {
        "model": label,
        "task": "image-text-to-text",
        "batch_mode": args.batch_mode,
        "batch_size": args.batch_size,
        "benchmark": {
            "prompt": args.prompt,
            "prefill_range": list(args.prefill_range),
            "cache_lengths": args.cache_lengths,
            "decode_window": args.decode_window,
            "llm_reference_resolution": llm_resolution,
            "vision_results": vision_results,
            "vision_summary": {
                "vision_encode_ms": _summary(all_vision_encode_ms),
                "vision_fps": _summary(all_vision_fps),
                "avg_power_w": _summary(all_vision_avg_power_w),
                "p99_power_w": _summary(all_vision_p99_power_w),
                "avg_utilization_pct": _summary(all_vision_avg_utilization_pct),
                "p99_utilization_pct": _summary(all_vision_p99_utilization_pct),
                "avg_temperature_c": _summary(all_vision_avg_temperature_c),
                "p99_temperature_c": _summary(all_vision_p99_temperature_c),
                "avg_memory_used_mb": _summary(all_vision_avg_memory_used_mb),
                "p99_memory_used_mb": _summary(all_vision_p99_memory_used_mb),
                "avg_memory_used_pct": _summary(all_vision_avg_memory_used_pct),
                "p99_memory_used_pct": _summary(all_vision_p99_memory_used_pct),
                "total_energy_j": _summary(all_vision_total_energy_j),
                "vision_img_per_j": _summary(all_vision_img_per_j),
                "vision_j_per_img": _summary(all_vision_j_per_img),
            },
            "llm_results": {
                "runs": [asdict(r) for r in llm_runs],
                "device_time_series_runs": llm_device_time_series_runs,
                "summary": {
                    "llm_prefill_tps": _summary(llm_prefill),
                    "llm_decode_tps": _summary(llm_decode),
                    "llm_ttft_ms": _summary(llm_ttft_ms),
                    "llm_decode_duration_ms": _summary(llm_decode_ms),
                    "llm_total_ms": _summary(llm_total_ms),
                    "llm_prefill_npu_latency_pct": _summary(llm_prefill_npu_pct),
                    "llm_decode_npu_latency_pct": _summary(llm_decode_npu_pct),
                    "avg_power_w": _summary(llm_avg_power_w),
                    "p99_power_w": _summary(llm_p99_power_w),
                    "avg_utilization_pct": _summary(llm_avg_utilization_pct),
                    "p99_utilization_pct": _summary(llm_p99_utilization_pct),
                    "avg_temperature_c": _summary(llm_avg_temperature_c),
                    "p99_temperature_c": _summary(llm_p99_temperature_c),
                    "avg_memory_used_mb": _summary(llm_avg_memory_used_mb),
                    "p99_memory_used_mb": _summary(llm_p99_memory_used_mb),
                    "avg_memory_used_pct": _summary(llm_avg_memory_used_pct),
                    "p99_memory_used_pct": _summary(llm_p99_memory_used_pct),
                    "total_energy_j": _summary(llm_total_energy_j),
                    "prefill_tok_per_j": _summary(llm_prefill_tok_per_j),
                    "decode_tok_per_j": _summary(llm_decode_tok_per_j),
                    "prefill_j_per_tok": _summary(llm_prefill_j_per_tok),
                    "decode_j_per_tok": _summary(llm_decode_j_per_tok),
                },
            },
        },
        "device": {
            "avg_power_w": _mean(llm_avg_power_w),
            "p99_power_w": max(llm_p99_power_w) if llm_p99_power_w else None,
            "avg_utilization_pct": _mean(llm_avg_utilization_pct),
            "p99_utilization_pct": max(llm_p99_utilization_pct) if llm_p99_utilization_pct else None,
            "avg_temperature_c": _mean(llm_avg_temperature_c),
            "p99_temperature_c": max(llm_p99_temperature_c) if llm_p99_temperature_c else None,
            "avg_memory_used_mb": _mean(llm_avg_memory_used_mb),
            "p99_memory_used_mb": max(llm_p99_memory_used_mb) if llm_p99_memory_used_mb else None,
            "avg_memory_used_pct": _mean(llm_avg_memory_used_pct),
            "p99_memory_used_pct": max(llm_p99_memory_used_pct) if llm_p99_memory_used_pct else None,
            "total_energy_j": _mean(llm_total_energy_j),
            "prefill_tps_last": llm_prefill[-1] if llm_prefill else None,
            "decode_tps_last": llm_decode[-1] if llm_decode else None,
            "prefill_tok_per_j_last": llm_prefill_tok_per_j[-1] if llm_prefill_tok_per_j else None,
            "decode_tok_per_j_last": llm_decode_tok_per_j[-1] if llm_decode_tok_per_j else None,
            "prefill_j_per_tok_last": llm_prefill_j_per_tok[-1] if llm_prefill_j_per_tok else None,
            "decode_j_per_tok_last": llm_decode_j_per_tok[-1] if llm_decode_j_per_tok else None,
            "vision_avg_power_w": _mean(all_vision_avg_power_w),
            "vision_p99_power_w": max(all_vision_p99_power_w) if all_vision_p99_power_w else None,
            "vision_avg_temperature_c": _mean(all_vision_avg_temperature_c),
            "vision_p99_temperature_c": max(all_vision_p99_temperature_c) if all_vision_p99_temperature_c else None,
            "vision_img_per_j": _mean(all_vision_img_per_j),
            "vision_j_per_img": _mean(all_vision_j_per_img),
        },
    }
    return payload, csv_rows


def _rebuild_combined(results_dir: Path) -> None:
    llm_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    vision_rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("benchmark_type") == "measure" or path.name == _HOST_PC_INFO_FILENAME:
            continue
        if not isinstance(payload.get("model"), str) or not isinstance(payload.get("benchmark"), dict):
            continue
        bench = payload.get("benchmark", {})
        summ = bench.get("llm_results", {}).get("summary", {})
        vision_summ = bench.get("vision_summary", {})
        llm_rows.append(
            {
                "model": payload.get("model"),
                "batch_mode": payload.get("batch_mode"),
                "batch_size": payload.get("batch_size"),
                "llm_reference_resolution": bench.get("llm_reference_resolution"),
                "llm_prefill_tps_mean": summ.get("llm_prefill_tps", {}).get("mean"),
                "llm_decode_tps_mean": summ.get("llm_decode_tps", {}).get("mean"),
                "llm_ttft_ms_mean": summ.get("llm_ttft_ms", {}).get("mean"),
                "llm_decode_duration_ms_mean": summ.get("llm_decode_duration_ms", {}).get("mean"),
                "llm_total_ms_mean": summ.get("llm_total_ms", {}).get("mean"),
                "llm_prefill_npu_latency_pct_mean": summ.get("llm_prefill_npu_latency_pct", {}).get("mean"),
                "llm_decode_npu_latency_pct_mean": summ.get("llm_decode_npu_latency_pct", {}).get("mean"),
                "llm_prefill_tok_per_j_mean": summ.get("prefill_tok_per_j", {}).get("mean"),
                "llm_decode_tok_per_j_mean": summ.get("decode_tok_per_j", {}).get("mean"),
                "llm_prefill_j_per_tok_mean": summ.get("prefill_j_per_tok", {}).get("mean"),
                "llm_decode_j_per_tok_mean": summ.get("decode_j_per_tok", {}).get("mean"),
                "avg_power_w": payload.get("device", {}).get("avg_power_w"),
                "avg_utilization_pct": payload.get("device", {}).get("avg_utilization_pct"),
                "avg_temperature_c": payload.get("device", {}).get("avg_temperature_c"),
                "p99_temperature_c": payload.get("device", {}).get("p99_temperature_c"),
                "avg_memory_used_mb": payload.get("device", {}).get("avg_memory_used_mb"),
                "total_energy_j": payload.get("device", {}).get("total_energy_j"),
                "vision_encode_ms_mean": vision_summ.get("vision_encode_ms", {}).get("mean"),
                "vision_fps_mean": vision_summ.get("vision_fps", {}).get("mean"),
                "vision_img_per_j_mean": vision_summ.get("vision_img_per_j", {}).get("mean"),
                "vision_j_per_img_mean": vision_summ.get("vision_j_per_img", {}).get("mean"),
            }
        )
        device = payload.get("device")
        if isinstance(device, dict):
            device_rows.append({"model": payload.get("model"), **device})
        for row in bench.get("vision_results", []):
            if not isinstance(row, dict):
                continue
            s = row.get("summary", {})
            vision_rows.append(
                {
                    "model": payload.get("model"),
                    "batch_mode": payload.get("batch_mode"),
                    "batch_size": payload.get("batch_size"),
                    "image_resolution": row.get("image_resolution"),
                    "vision_encode_ms_mean": s.get("vision_encode_ms", {}).get("mean"),
                    "vision_fps_mean": s.get("vision_fps", {}).get("mean"),
                    "vision_img_per_j_mean": s.get("vision_img_per_j", {}).get("mean"),
                    "vision_j_per_img_mean": s.get("vision_j_per_img", {}).get("mean"),
                }
            )
    _write_csv(results_dir / "combined.csv", llm_rows)
    _write_csv(results_dir / "combined_llm.csv", llm_rows)
    _write_csv(results_dir / "combined_vision.csv", vision_rows)
    _write_csv(results_dir / "combined_device.csv", device_rows)
    if not llm_rows:
        _write_vlm_summary(results_dir)
        return
    with (results_dir / "combined.md").open("w", encoding="utf-8") as f:
        headers = list(llm_rows[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |\n")
        for row in llm_rows:
            vals = [str(row.get(h, "")) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")
    models = [str(r["model"]) for r in llm_rows]
    prefill = [float(r.get("llm_prefill_tps_mean") or 0.0) for r in llm_rows]
    decode = [float(r.get("llm_decode_tps_mean") or 0.0) for r in llm_rows]
    avg_power = [float(r.get("avg_power_w") or 0.0) for r in llm_rows]
    avg_temp = [float(r.get("avg_temperature_c") or 0.0) for r in llm_rows]
    avg_util = [float(r.get("avg_utilization_pct") or 0.0) for r in llm_rows]
    avg_mem_mb = [float(r.get("avg_memory_used_mb") or 0.0) for r in llm_rows]
    total_energy = [float(r.get("total_energy_j") or 0.0) for r in llm_rows]
    llm_prefill_tpj = [float(r.get("llm_prefill_tok_per_j_mean") or 0.0) for r in llm_rows]
    llm_decode_tpj = [float(r.get("llm_decode_tok_per_j_mean") or 0.0) for r in llm_rows]
    chart_specs = [
        ("llm_prefill_tps.png", prefill, "Tokens Per Second", "Prefill Tokens Per Second"),
        ("llm_prefill_tokens_per_j.png", llm_prefill_tpj, "Tokens Per Joule", "Prefill Tokens Per Joule"),
        ("llm_decode_tps.png", decode, "Tokens Per Second", "Decode Tokens Per Second"),
        ("llm_decode_tokens_per_j.png", llm_decode_tpj, "Tokens Per Joule", "Decode Tokens Per Joule"),
        ("avg_power_w.png", avg_power, "Power (Watts)", "Power"),
        ("avg_temperature_c.png", avg_temp, "Temperature (Celsius)", "Temperature"),
        ("avg_utilization_pct.png", avg_util, "Utilization (Percent)", "Utilization"),
        ("avg_memory_used_mb.png", avg_mem_mb, "Memory Used (Megabytes)", "Memory Used Megabytes"),
        ("total_energy_j.png", total_energy, "Energy (Joules)", "Total Energy"),
    ]
    for filename, values, x_label, title in chart_specs:
        plot_simple_barh(
            labels=models,
            values=values,
            x_label=x_label,
            title=title,
            output_path=results_dir / filename,
        )
    _write_vlm_summary(results_dir)


def _write_vlm_summary(results_dir: Path, *, measure: bool = False) -> None:
    """Write an image-text-to-text benchmark summary Markdown with host info, plots, and table."""
    table_name = "combined_measure.md" if measure else "combined.md"
    summary_name = "summary_measure.md" if measure else "summary.md"
    title = "Image Text-to-Text Measure Benchmark Summary" if measure else "Image Text-to-Text Benchmark Summary"
    prefixes = ("measure_",) if measure else None
    _write_summary_markdown(
        results_dir / summary_name,
        title=title,
        host_info_path=results_dir / _HOST_PC_INFO_FILENAME,
        table_markdown_path=results_dir / table_name,
        plot_paths=_existing_png_paths(results_dir, prefixes=prefixes),
    )


def _plot_model(payload: dict[str, Any], output_path: Path) -> None:
    bench = payload.get("benchmark", {})
    vision = bench.get("vision_results", [])
    llm = bench.get("llm_results", {})

    resolutions = [int(x.get("image_resolution", 0)) for x in vision]
    encode_ms = [float(x.get("summary", {}).get("vision_encode_ms", {}).get("mean", 0.0)) for x in vision]
    fps = [float(x.get("summary", {}).get("vision_fps", {}).get("mean", 0.0)) for x in vision]
    prefill = float(llm.get("summary", {}).get("llm_prefill_tps", {}).get("mean", 0.0))
    decode = float(llm.get("summary", {}).get("llm_decode_tps", {}).get("mean", 0.0))
    ttft = float(llm.get("summary", {}).get("llm_ttft_ms", {}).get("mean", 0.0))

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(resolutions, encode_ms, "o-", color="tab:red")
    axs[0].set_title("Vision Encode ms")
    axs[0].set_xlabel("resolution")
    axs[0].grid(True, alpha=0.3)
    axs[1].plot(resolutions, fps, "o-", color="tab:blue")
    axs[1].set_title("Vision FPS")
    axs[1].set_xlabel("resolution")
    axs[1].grid(True, alpha=0.3)
    axs[2].bar(
        ["prefill_tps", "decode_tps", "ttft_ms"],
        [prefill, decode, ttft],
        color=["tab:green", "tab:purple", "tab:orange"],
    )
    axs[2].set_title("LLM Summary")
    axs[2].grid(axis="y", alpha=0.3)
    fig.suptitle(payload.get("model", "unknown"))
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by image-text-to-text benchmark subcommands."""
    _add_pipeline_device_args(parser, device_default=None, trust_remote_code_default=True)
    _add_batch_selection_args(parser)
    _add_device_tracking_args(parser)
    parser.add_argument("--model", default=None, help="single model id to benchmark (optional)")
    parser.add_argument("--tokenizer", default=None, help="tokenizer id or local path (optional)")
    parser.add_argument("--revision", default=None, help="model revision (e.g., W8)")
    parser.add_argument("--all", action="store_true", help="benchmark W8 and W4V8 revisions only (skip main)")
    parser.add_argument("--mxq-path", default=None, help="override mxq_path for pipeline loading")
    parser.add_argument(
        "--mxq-dir",
        default=None,
        help="directory containing local mxq files. When set, files matching <model_id>-<W8|W4V8>.mxq are benchmarked.",
    )
    parser.add_argument(
        "--core-mode",
        choices=[*list(_CORE_MODE_CHOICES_COMMON), "all"],
        default="global8",
        help="core mode passed to model_kwargs; all expands to single/global4/global8 (default: global8)",
    )
    parser.add_argument("--prompt", default="Describe the image in one sentence.")
    parser.add_argument(
        "--warmup", type=_parse_positive_int, default=1, help="number of warmup runs before measured runs"
    )
    parser.add_argument("--repeat", type=_parse_positive_int, default=1, help="number of repeated runs")
    parser.add_argument(
        "--prefill-chunk-size",
        type=_parse_positive_int_optional,
        default=None,
        help="optional prefill_chunk_size forwarded to the VLM LLM benchmark",
    )
    parser.add_argument(
        "--original-models",
        action="store_true",
        help="resolve each Mobilint model to parent/base model id from HF Hub",
    )
    parser.add_argument(
        "--cuda-precheck",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="best-effort CUDA VRAM pre-check before loading each model",
    )
    parser.add_argument(
        "--cuda-precheck-margin",
        type=float,
        default=1.15,
        help="required free VRAM factor versus estimated model weights (default: 1.15)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="skip models with existing outputs")
    parser.add_argument("--rebuild-charts", action="store_true", help="skip benchmark run and rebuild combined outputs")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="output directory (default: benchmark/transformers/results/image_text_to_text)",
    )


def _add_measure_args(parser: argparse.ArgumentParser) -> None:
    """Add TPS measure-aligned VLM fixed measurement arguments."""
    parser.add_argument(
        "--image-resolution", type=_parse_positive_int, default=224, help="image resolution (default: 224)"
    )
    parser.add_argument(
        "--prefill", type=_parse_positive_int, default=128, help="LLM prefill token count (default: 128)"
    )
    parser.add_argument("--decode", type=_parse_positive_int, default=32, help="LLM decode token count (default: 32)")


def _add_sweep_args(parser: argparse.ArgumentParser) -> None:
    """Add TPS sweep-aligned VLM grid measurement arguments."""
    parser.add_argument(
        "--image-resolutions",
        type=_parse_int_csv,
        default=_parse_int_csv("224,384,512,768"),
        help="comma-separated image resolutions (default: 224,384,512,768)",
    )
    parser.add_argument(
        "--llm-resolution",
        type=_parse_positive_int_optional,
        default=None,
        help="reference resolution used for LLM-only benchmark (default: first image resolution)",
    )
    parser.add_argument(
        "--prefill-range",
        type=_parse_range_arg,
        default=(512, 2048, 512),
        help="LLM prefill sweep range by total multimodal prefix length (default: 512:2048:512)",
    )
    parser.add_argument(
        "--cache-lengths",
        type=_parse_int_csv,
        default=_parse_int_csv("128,512,1024,2048"),
        help="comma-separated LLM cache lengths for decode sweep (default: 128,512,1024,2048)",
    )
    parser.add_argument(
        "--decode-window",
        type=_parse_positive_int,
        default=32,
        help="decode token window for LLM cache-length sweep (default: 32)",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the image-text-to-text benchmark argument parser."""
    parser = argparse.ArgumentParser(description="Benchmark image-text-to-text models.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure", help="measure fixed image/prefill/decode TPS")
    _add_common_benchmark_args(measure)
    _add_measure_args(measure)
    measure.set_defaults(_handler=_run_measure)
    sweep = subparsers.add_parser("sweep", help="run image and LLM TPS sweeps")
    _add_common_benchmark_args(sweep)
    _add_sweep_args(sweep)
    sweep.set_defaults(_handler=_run_sweep)
    return parser


def _flag_present(raw_argv: list[str], flag: str) -> bool:
    """Return whether a flag appears in raw argv."""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in raw_argv)


def _resolve_batch_core_mode(args: argparse.Namespace, *, core_mode_explicit: bool) -> None:
    """Apply the single-core constraint for batch LLM benchmarks."""
    if args.batch_mode != "batch":
        return
    if core_mode_explicit and args.core_mode != "single":
        raise SystemExit("--batch only supports --core-mode single for batch LLM benchmarks.")
    args.core_mode = "single"


def _resolve_runtime_defaults(args: argparse.Namespace, raw_argv: list[str]) -> None:
    """Apply benchmark runtime defaults that depend on explicit CLI flags."""
    device_explicit = _flag_present(raw_argv, "--device")
    device_backend_explicit = _flag_present(raw_argv, "--device-backend")
    core_mode_explicit = _flag_present(raw_argv, "--core-mode")
    args._device_backend_explicit = device_backend_explicit
    args._device_backend_requested = args.device_backend
    args._core_mode_explicit = core_mode_explicit
    args.device = _resolve_default_device_common(
        device=args.device,
        device_explicit=device_explicit,
        model_id=args.model,
        mxq_path=args.mxq_path,
        mxq_dir=args.mxq_dir,
        original_models=args.original_models,
    )
    if not device_explicit:
        print(f"Auto-set --device={args.device}")
    args.device_backend = _resolve_default_device_backend_common(
        device_backend=args.device_backend,
        device_backend_explicit=device_backend_explicit,
        model_id=args.model,
        mxq_path=args.mxq_path,
        mxq_dir=args.mxq_dir,
        original_models=args.original_models,
    )
    if not device_backend_explicit:
        if args.model or args.mxq_path or args.mxq_dir:
            print(f"Auto-set --device-backend={args.device_backend} (based on target/device policy)")
        else:
            print("Auto-set --device-backend per target (based on target/device policy)")
    _resolve_batch_core_mode(args, core_mode_explicit=core_mode_explicit)


def _args_for_target_device_backend(
    args: argparse.Namespace,
    *,
    model_id: str,
    mxq_path: str | None = None,
) -> argparse.Namespace:
    """Return an args copy with a device backend resolved for one benchmark target."""
    resolved = copy.copy(args)
    requested_backend = getattr(args, "_device_backend_requested", args.device_backend)
    resolved.device_backend = _resolve_default_device_backend_common(
        device_backend=requested_backend,
        device_backend_explicit=bool(getattr(args, "_device_backend_explicit", False)),
        model_id=model_id,
        mxq_path=mxq_path,
        mxq_dir=args.mxq_dir,
        original_models=args.original_models,
    )
    return resolved


def main(argv: list[str] | None = None) -> int:
    """Run the selected image-text-to-text benchmark subcommand."""
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _resolve_runtime_defaults(args, raw_argv)
    return args._handler(args)


def _run_sweep(args: argparse.Namespace) -> int:
    """Run multi-model image-text-to-text sweep benchmarks."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    disable_npu_specific_args = bool(args.original_models and not args.mxq_dir)
    if disable_npu_specific_args:
        print("Note: --original-models is enabled; skipping NPU-specific parameters (core_mode/prefill_chunk_size).")
    _resolve_batch_core_mode(args, core_mode_explicit=bool(getattr(args, "_core_mode_explicit", False)))
    script_dir = Path(__file__).resolve().parent
    results_dir = (
        Path(args.results_dir).resolve() if args.results_dir else script_dir / "results" / "image_text_to_text"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_charts:
        _rebuild_combined(results_dir)
        return 0

    _collect_host_pc_info(results_dir)

    available_model_ids = list_models(tasks="image-text-to-text").get("image-text-to-text", [])
    if not available_model_ids:
        print("No image-text-to-text models found.")
        return 0
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = []
    if args.mxq_dir:
        mxq_dir = Path(args.mxq_dir).expanduser().resolve()
        if not mxq_dir.is_dir():
            raise SystemExit(f"--mxq-dir is not a directory: {mxq_dir}")
        if args.model or args.original_models or args.all or args.revision:
            print(
                "Note: --mxq-dir is set, so --model/--original-models/--all/--revision are ignored "
                "(revision and mxq_path are taken from filename)."
            )
        for model_id, rev_candidates, label, base, target_mxq_path in _iter_targets_from_mxq_dir(
            mxq_dir=mxq_dir,
            available_model_ids=available_model_ids,
        ):
            raw_targets.append((model_id, rev_candidates, label, base, target_mxq_path))
    else:
        if args.model:
            model_ids = [str(args.model)]
        else:
            model_ids = available_model_ids
        if args.original_models:
            original_count = len(model_ids)
            model_ids = _resolve_original_model_ids(model_ids)
            print(
                f"Using parent/original model ids: {len(model_ids)} unique models "
                f"(from {original_count} listed models)."
            )
        for model_id, revision, label, base in _iter_targets(model_ids, args.revision, args.all):
            raw_targets.append((model_id, [revision], label, base, args.mxq_path))

    targets = [
        VLMBenchmarkTarget(
            model_id=target.model_id,
            revision=target.revision_candidates[0] if target.revision_candidates else None,
            label=target.label,
            base=target.base,
            mxq_path=target.mxq_path,
            max_batch_size=target.max_batch_size,
        )
        for target in _filter_text_targets_by_batch_mode(
            raw_targets,
            batch_mode=args.batch_mode,
            task="image-text-to-text",
        )
    ]
    core_modes = [None] if disable_npu_specific_args else _iter_core_modes_common(args.core_mode)
    run_targets: list[tuple[str, str | None, str, str, str | None, str | None, int]] = []
    for target in targets:
        for core_mode in core_modes:
            mode_label, mode_base = _append_core_mode_suffix_common(target.label, target.base, core_mode)
            run_targets.append(
                (
                    target.model_id,
                    target.revision,
                    mode_label,
                    mode_base,
                    target.mxq_path,
                    core_mode,
                    target.max_batch_size,
                )
            )

    for model_id, revision, label, base, target_mxq_path, core_mode, batch_size in tqdm(
        run_targets,
        desc="Benchmarking VLM models",
        unit="model-mode",
    ):
        target_args = _args_for_target_device_backend(args, model_id=model_id, mxq_path=target_mxq_path)
        if _is_cuda_device(args.device):
            _clear_cuda_memory(args.device)
        json_path = results_dir / f"{base}.json"
        csv_path = results_dir / f"{base}.csv"
        png_path = results_dir / f"{base}.png"
        if args.skip_existing and json_path.is_file() and csv_path.is_file() and png_path.is_file():
            print(f"Skipping {label} (results exist).")
            continue
        artifacts_available, skip_reason = _vlm_revision_artifacts_available(model_id, revision, target_mxq_path)
        if not artifacts_available:
            print(f"Skipping {label} ({skip_reason}).")
            continue
        if _should_precheck_cuda(args):
            estimated = _estimate_model_weight_bytes(model_id, revision)
            mem_info = _cuda_memory_info(args.device)
            if estimated is not None and mem_info is not None:
                free_b, _ = mem_info
                required = int(float(estimated) * float(args.cuda_precheck_margin))
                if free_b < required:
                    print(
                        "Skipping (pre-check VRAM insufficient): "
                        f"free={_format_gib(free_b)} required~={_format_gib(required)} "
                        f"estimated_weights={_format_gib(estimated)}"
                    )
                    _clear_cuda_memory(args.device)
                    continue
        pipeline = None
        try:
            pipeline = _build_pipeline(args, model_id, revision, target_mxq_path, core_mode)
            target_args.batch_size = batch_size
            payload, rows = _run_model(target_args, label, pipeline)
            _write_json(json_path, payload)
            _write_csv(csv_path, rows)
            _plot_model(payload, png_path)
            print(f"Saved: {json_path.name}, {csv_path.name}, {png_path.name}")
        except Exception as e:
            if _is_cuda_oom_error(e):
                print(f"Skipping {label} (CUDA OOM): {e}")
            else:
                print(f"Skipping {label} (benchmark failed): {e}")
        finally:
            _release_pipeline(pipeline, args.device)

    _rebuild_combined(results_dir)
    return 0


def _collect_vlm_run_targets(
    args: argparse.Namespace,
) -> tuple[Path, bool, list[tuple[str, str | None, str, str, str | None, str | None, int]]]:
    """Resolve image-text-to-text benchmark targets and core-mode expansion."""
    script_dir = Path(__file__).resolve().parent
    results_dir = (
        Path(args.results_dir).resolve() if args.results_dir else script_dir / "results" / "image_text_to_text"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    available_model_ids = list_models(tasks="image-text-to-text").get("image-text-to-text", [])
    if not available_model_ids:
        print("No image-text-to-text models found.")
        return results_dir, False, []
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = []
    if args.mxq_dir:
        mxq_dir = Path(args.mxq_dir).expanduser().resolve()
        if not mxq_dir.is_dir():
            raise SystemExit(f"--mxq-dir is not a directory: {mxq_dir}")
        for model_id, rev_candidates, label, base, target_mxq_path in _iter_targets_from_mxq_dir(
            mxq_dir=mxq_dir,
            available_model_ids=available_model_ids,
        ):
            raw_targets.append((model_id, rev_candidates, label, base, target_mxq_path))
    else:
        model_ids = [str(args.model)] if args.model else available_model_ids
        if args.original_models:
            model_ids = _resolve_original_model_ids(model_ids)
        for model_id, revision, label, base in _iter_targets(model_ids, args.revision, args.all):
            raw_targets.append((model_id, [revision], label, base, args.mxq_path))
    disable_npu_specific_args = bool(args.original_models and not args.mxq_dir)
    _resolve_batch_core_mode(args, core_mode_explicit=bool(getattr(args, "_core_mode_explicit", False)))
    targets = [
        VLMBenchmarkTarget(
            model_id=target.model_id,
            revision=target.revision_candidates[0] if target.revision_candidates else None,
            label=target.label,
            base=target.base,
            mxq_path=target.mxq_path,
            max_batch_size=target.max_batch_size,
        )
        for target in _filter_text_targets_by_batch_mode(
            raw_targets,
            batch_mode=args.batch_mode,
            task="image-text-to-text",
        )
    ]
    core_modes = [None] if disable_npu_specific_args else _iter_core_modes_common(args.core_mode)
    run_targets: list[tuple[str, str | None, str, str, str | None, str | None, int]] = []
    for target in targets:
        for core_mode in core_modes:
            mode_label, mode_base = _append_core_mode_suffix_common(target.label, target.base, core_mode)
            run_targets.append(
                (
                    target.model_id,
                    target.revision,
                    mode_label,
                    mode_base,
                    target.mxq_path,
                    core_mode,
                    target.max_batch_size,
                )
            )
    return results_dir, disable_npu_specific_args, run_targets


def _collect_measure_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert VLM measure payloads to combined rows."""
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        summary = payload.get("summary", {})
        device = payload.get("device") or {}
        rows.append(
            {
                "model": payload.get("model"),
                "batch_mode": payload.get("batch_mode"),
                "batch_size": payload.get("batch_size"),
                "image_resolution": payload.get("image_resolution"),
                "prefill_tokens": payload.get("prefill"),
                "decode_tokens": payload.get("decode"),
                "repeat": payload.get("repeat"),
                "vision_encode_ms_mean": summary.get("vision_encode_ms", {}).get("mean"),
                "vision_fps_mean": summary.get("vision_fps", {}).get("mean"),
                "llm_prefill_tps_mean": summary.get("llm_prefill_tps", {}).get("mean"),
                "llm_decode_tps_mean": summary.get("llm_decode_tps", {}).get("mean"),
                "llm_ttft_ms_mean": summary.get("llm_ttft_ms", {}).get("mean"),
                "llm_decode_duration_ms_mean": summary.get("llm_decode_duration_ms", {}).get("mean"),
                "avg_power_w": device.get("avg_power_w"),
                "avg_temperature_c": device.get("avg_temperature_c"),
                "avg_utilization_pct": device.get("avg_utilization_pct"),
                "avg_memory_used_mb": device.get("avg_memory_used_mb"),
                "total_energy_j": device.get("total_energy_j"),
                "llm_prefill_tok_per_j_mean": device.get("llm_prefill_tok_per_j"),
                "llm_decode_tok_per_j_mean": device.get("llm_decode_tok_per_j"),
                "vision_img_per_j_mean": device.get("vision_img_per_j"),
            }
        )
    return rows


def _write_measure_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write VLM combined measure Markdown."""
    if not rows:
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |\n")
        for row in rows:
            f.write("| " + " | ".join("" if row.get(h) is None else str(row.get(h)) for h in headers) + " |\n")


def _rebuild_measure_outputs(results_dir: Path) -> None:
    """Rebuild VLM measure combined outputs."""
    payloads: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*_measure.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("benchmark_type") == "measure":
            payloads.append(payload)
    if not payloads:
        print("No measure JSON results found. Nothing to aggregate.")
        _write_vlm_summary(results_dir, measure=True)
        return
    rows = _collect_measure_rows(payloads)
    _write_csv(results_dir / "combined_measure.csv", rows)
    _write_measure_markdown(results_dir / "combined_measure.md", rows)
    models = [str(row["model"]) for row in rows]
    chart_specs = [
        ("measure_llm_prefill_tps.png", "llm_prefill_tps_mean", "Tokens Per Second", "Prefill Tokens Per Second"),
        (
            "measure_llm_prefill_tokens_per_j.png",
            "llm_prefill_tok_per_j_mean",
            "Tokens Per Joule",
            "Prefill Tokens Per Joule",
        ),
        ("measure_llm_decode_tps.png", "llm_decode_tps_mean", "Tokens Per Second", "Decode Tokens Per Second"),
        (
            "measure_llm_decode_tokens_per_j.png",
            "llm_decode_tok_per_j_mean",
            "Tokens Per Joule",
            "Decode Tokens Per Joule",
        ),
        ("measure_avg_power_w.png", "avg_power_w", "Power (Watts)", "Power"),
        ("measure_avg_temperature_c.png", "avg_temperature_c", "Temperature (Celsius)", "Temperature"),
        ("measure_avg_utilization_pct.png", "avg_utilization_pct", "Utilization (Percent)", "Utilization"),
        ("measure_avg_memory_used_mb.png", "avg_memory_used_mb", "Memory Used (Megabytes)", "Memory Used Megabytes"),
        ("measure_total_energy_j.png", "total_energy_j", "Energy (Joules)", "Total Energy"),
    ]
    for filename, key, x_label, title in chart_specs:
        plot_simple_barh(
            labels=models,
            values=[float(row.get(key) or 0.0) for row in rows],
            x_label=x_label,
            title=title,
            output_path=results_dir / filename,
        )
    _write_vlm_summary(results_dir, measure=True)


def _resolve_vlm_results_dir(args: argparse.Namespace) -> Path:
    """Resolve and create the image-text-to-text benchmark results directory."""
    script_dir = Path(__file__).resolve().parent
    results_dir = (
        Path(args.results_dir).resolve() if args.results_dir else script_dir / "results" / "image_text_to_text"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _run_measure(args: argparse.Namespace) -> int:
    """Run multi-model image-text-to-text fixed image/prefill/decode benchmarks."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.rebuild_charts:
        _rebuild_measure_outputs(_resolve_vlm_results_dir(args))
        return 0
    results_dir, disable_npu_specific_args, run_targets = _collect_vlm_run_targets(args)
    if not run_targets:
        return 0
    _collect_host_pc_info(results_dir)
    for model_id, revision, label, base, target_mxq_path, core_mode, batch_size in tqdm(
        run_targets, desc="Measuring VLM models", unit="model-mode"
    ):
        target_args = _args_for_target_device_backend(args, model_id=model_id, mxq_path=target_mxq_path)
        target_args.batch_size = batch_size
        if _is_cuda_device(args.device):
            _clear_cuda_memory(args.device)
        json_path = results_dir / f"{base}_measure.json"
        if args.skip_existing and json_path.is_file():
            print(f"Skipping {label} (measure result exists).")
            continue
        artifacts_available, skip_reason = _vlm_revision_artifacts_available(model_id, revision, target_mxq_path)
        if not artifacts_available:
            print(f"Skipping {label} ({skip_reason}).")
            continue
        pipeline = None
        try:
            pipeline = _build_pipeline(args, model_id, revision, target_mxq_path, core_mode)
            measurer = VLMTPSMeasurer(pipeline)
            tracker = _build_device_tracker(target_args, pipeline)
            _print_device_status(target_args, tracker)
            resolved_prefill_chunk_size = None if disable_npu_specific_args else args.prefill_chunk_size
            for warmup_idx in tqdm(range(args.warmup), desc=f"{label} warmup", leave=False):
                measurer.measure_vision(
                    args.image_resolution,
                    repeat=1,
                    prompt=args.prompt,
                    batch_size=batch_size,
                    show_progress=False,
                )
                measurer.measure_llm_full(
                    image_resolution=args.image_resolution,
                    prompt=args.prompt,
                    prefill_range=(args.prefill, args.prefill, args.prefill),
                    cache_lengths=[args.prefill],
                    decode_window=args.decode,
                    prefill_chunk_size=resolved_prefill_chunk_size,
                    batch_size=batch_size,
                    show_progress=True,
                    progress_prefix=f"{label} warmup {warmup_idx + 1}/{args.warmup}",
                )
            vision_runs: list[dict[str, float]] = []
            llm_runs: list[dict[str, Any]] = []
            device_time_series_runs: list[dict[str, list[dict[str, float]]]] = []
            avg_power_w: list[float] = []
            avg_temperature_c: list[float] = []
            avg_utilization_pct: list[float] = []
            avg_memory_used_mb: list[float] = []
            total_energy_j: list[float] = []
            llm_prefill_tok_per_j: list[float] = []
            llm_decode_tok_per_j: list[float] = []
            for repeat_idx in tqdm(range(args.repeat), desc=f"{label} measured runs", leave=False):
                if tracker is not None:
                    tracker.start()
                vision_latency, vision_fps = measurer.measure_vision(
                    args.image_resolution,
                    repeat=1,
                    prompt=args.prompt,
                    batch_size=batch_size,
                    show_progress=False,
                )[0]
                try:
                    llm_result = measurer.measure_llm_full(
                        image_resolution=args.image_resolution,
                        prompt=args.prompt,
                        prefill_range=(args.prefill, args.prefill, args.prefill),
                        cache_lengths=[args.prefill],
                        decode_window=args.decode,
                        prefill_chunk_size=resolved_prefill_chunk_size,
                        batch_size=batch_size,
                        show_progress=True,
                        progress_prefix=f"{label} run {repeat_idx + 1}/{args.repeat}",
                    )
                finally:
                    _stop_tracker_safe(tracker)
                vision_runs.append({"vision_encode_latency": vision_latency, "vision_fps": vision_fps})
                llm_runs.append(asdict(llm_result))
                if tracker is not None:
                    metric = _extract_device_metric(tracker)
                    device_time_series_runs.append(_extract_device_time_series(tracker))
                    power = metric.get("avg_power_w")
                    temperature = metric.get("avg_temperature_c")
                    utilization = metric.get("avg_utilization_pct")
                    memory_used = metric.get("avg_memory_used_mb")
                    if temperature is not None:
                        avg_temperature_c.append(float(temperature))
                    if utilization is not None:
                        avg_utilization_pct.append(float(utilization))
                    if memory_used is not None:
                        avg_memory_used_mb.append(float(memory_used))
                    if power is not None:
                        total_time = vision_latency + float(llm_result.prefill_phase_duration_s or 0.0)
                        total_time += float(llm_result.decode_phase_duration_s or 0.0)
                        avg_power_w.append(float(power))
                        total_energy_j.append(float(power) * total_time)
            llm_prefill = [
                float(r["prefill_sweep"]["tps_values"][-1]) for r in llm_runs if r["prefill_sweep"]["tps_values"]
            ]
            llm_decode = [
                float(r["decode_sweep"]["tps_values"][-1]) for r in llm_runs if r["decode_sweep"]["tps_values"]
            ]
            llm_ttft = [
                float(r["prefill_sweep"]["time_values"][-1]) * 1000.0
                for r in llm_runs
                if r["prefill_sweep"]["time_values"]
            ]
            llm_decode_ms = [
                float(r["decode_sweep"]["time_values"][-1]) * 1000.0
                for r in llm_runs
                if r["decode_sweep"]["time_values"]
            ]
            avg_power = _mean(avg_power_w)
            if avg_power > 0.0:
                llm_prefill_tok_per_j = [value / avg_power for value in llm_prefill]
                llm_decode_tok_per_j = [value / avg_power for value in llm_decode]
            payload = {
                "model": label,
                "benchmark_type": "measure",
                "task": "image-text-to-text",
                "batch_mode": args.batch_mode,
                "batch_size": batch_size,
                "prompt": args.prompt,
                "image_resolution": args.image_resolution,
                "prefill": args.prefill,
                "decode": args.decode,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "vision_runs": vision_runs,
                "llm_runs": llm_runs,
                "summary": {
                    "vision_encode_ms": _summary([r["vision_encode_latency"] * 1000.0 for r in vision_runs]),
                    "vision_fps": _summary([r["vision_fps"] for r in vision_runs]),
                    "llm_prefill_tps": _summary(llm_prefill),
                    "llm_decode_tps": _summary(llm_decode),
                    "llm_ttft_ms": _summary(llm_ttft),
                    "llm_decode_duration_ms": _summary(llm_decode_ms),
                },
                "device": {
                    "avg_power_w": avg_power,
                    "avg_temperature_c": _mean(avg_temperature_c),
                    "avg_utilization_pct": _mean(avg_utilization_pct),
                    "avg_memory_used_mb": _mean(avg_memory_used_mb),
                    "total_energy_j": sum(total_energy_j) if total_energy_j else None,
                    "llm_prefill_tok_per_j": _mean(llm_prefill_tok_per_j),
                    "llm_decode_tok_per_j": _mean(llm_decode_tok_per_j),
                    "vision_img_per_j": _safe_div(len(vision_runs), sum(total_energy_j)) if total_energy_j else None,
                }
                if avg_power_w or total_energy_j
                else None,
                "device_time_series_runs": device_time_series_runs,
            }
            _write_json(json_path, payload)
            print(f"Saved: {json_path.name}")
        except Exception as e:
            print(f"Skipping {label} (measure failed): {e}")
        finally:
            _release_pipeline(pipeline, args.device)
    _rebuild_measure_outputs(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
