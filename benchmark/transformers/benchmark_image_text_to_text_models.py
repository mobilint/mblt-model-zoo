from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
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
        _cuda_memory_info,
        _estimate_model_weight_bytes,
        _format_gib,
        _is_cuda_oom_error,
        _iter_targets_from_mxq_dir,
        _resolve_original_model_ids,
        _should_precheck_cuda,
    )
except Exception:
    from .benchmark_text_generation_models import (
        _cuda_memory_info,
        _estimate_model_weight_bytes,
        _format_gib,
        _is_cuda_oom_error,
        _iter_targets_from_mxq_dir,
        _resolve_original_model_ids,
        _should_precheck_cuda,
    )


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
    model_kwargs = _apply_core_mode_model_kwargs_common(model_kwargs, core_mode)
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
            measurer.measure_vision(image_resolution=resolution, repeat=1, prompt=args.prompt, show_progress=False)
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
                    image_resolution=resolution, repeat=1, prompt=args.prompt, show_progress=False
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
    for _ in tqdm(
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
            show_progress=False,
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
    for _ in tqdm(
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
                show_progress=False,
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
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("benchmark_type") == "measure":
            continue
        bench = payload.get("benchmark", {})
        summ = bench.get("llm_results", {}).get("summary", {})
        vision_summ = bench.get("vision_summary", {})
        llm_rows.append(
            {
                "model": payload.get("model"),
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
                "avg_temperature_c": payload.get("device", {}).get("avg_temperature_c"),
                "p99_temperature_c": payload.get("device", {}).get("p99_temperature_c"),
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
    ttft = [float(r.get("llm_ttft_ms_mean") or 0.0) for r in llm_rows]
    vision_encode = [float(r.get("vision_encode_ms_mean") or 0.0) for r in llm_rows]
    vision_fps = [float(r.get("vision_fps_mean") or 0.0) for r in llm_rows]
    avg_temp = [float(r.get("avg_temperature_c") or 0.0) for r in llm_rows]
    p99_temp = [float(r.get("p99_temperature_c") or 0.0) for r in llm_rows]
    llm_prefill_tpj = [float(r.get("llm_prefill_tok_per_j_mean") or 0.0) for r in llm_rows]
    llm_decode_tpj = [float(r.get("llm_decode_tok_per_j_mean") or 0.0) for r in llm_rows]
    llm_prefill_jpt = [float(r.get("llm_prefill_j_per_tok_mean") or 0.0) for r in llm_rows]
    llm_decode_jpt = [float(r.get("llm_decode_j_per_tok_mean") or 0.0) for r in llm_rows]
    vision_img_per_j = [float(r.get("vision_img_per_j_mean") or 0.0) for r in llm_rows]
    vision_j_per_img = [float(r.get("vision_j_per_img_mean") or 0.0) for r in llm_rows]
    chart_specs = [
        ("llm_decode_tps.png", decode, "tok/s", "LLM Decode TPS (mean)"),
        ("llm_prefill_tps.png", prefill, "tok/s", "LLM Prefill TPS (mean)"),
        ("llm_ttft_ms.png", ttft, "ms", "LLM TTFT (mean)"),
        ("vision_encode_ms.png", vision_encode, "ms", "Vision Encode Latency (mean)"),
        ("vision_fps.png", vision_fps, "fps", "Vision FPS (mean)"),
        ("avg_temperature_c.png", avg_temp, "C", "Average Temperature"),
        ("p99_temperature_c.png", p99_temp, "C", "P99 Temperature"),
        ("llm_prefill_tokens_per_j.png", llm_prefill_tpj, "tok/J", "LLM Prefill Tokens/J (mean)"),
        ("llm_decode_tokens_per_j.png", llm_decode_tpj, "tok/J", "LLM Decode Tokens/J (mean)"),
        ("llm_prefill_j_per_token.png", llm_prefill_jpt, "J/tok", "LLM Prefill J/Token (mean)"),
        ("llm_decode_j_per_token.png", llm_decode_jpt, "J/tok", "LLM Decode J/Token (mean)"),
        ("vision_img_per_j.png", vision_img_per_j, "img/J", "Vision Img/J (mean)"),
        ("vision_j_per_img.png", vision_j_per_img, "J/img", "Vision J/Img (mean)"),
    ]
    for filename, values, x_label, title in chart_specs:
        plot_simple_barh(
            labels=models,
            values=values,
            x_label=x_label,
            title=title,
            output_path=results_dir / filename,
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
        default=None,
        help="core mode passed to model_kwargs; all expands to single/global4/global8 (default: None)",
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


def _resolve_runtime_defaults(args: argparse.Namespace, raw_argv: list[str]) -> None:
    """Apply benchmark runtime defaults that depend on explicit CLI flags."""
    device_explicit = _flag_present(raw_argv, "--device")
    device_backend_explicit = _flag_present(raw_argv, "--device-backend")
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
    if not device_backend_explicit:
        args.device_backend = _resolve_default_device_backend_common(
            device_backend=args.device_backend,
            device_backend_explicit=device_backend_explicit,
            model_id=args.model,
            mxq_path=args.mxq_path,
            mxq_dir=args.mxq_dir,
            original_models=args.original_models,
        )
        print(f"Auto-set --device-backend={args.device_backend} (based on target/device policy)")


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
    script_dir = Path(__file__).resolve().parent
    results_dir = (
        Path(args.results_dir).resolve() if args.results_dir else script_dir / "results" / "image_text_to_text"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_charts:
        _rebuild_combined(results_dir)
        return 0

    available_model_ids = list_models(tasks="image-text-to-text").get("image-text-to-text", [])
    if not available_model_ids:
        print("No image-text-to-text models found.")
        return 0
    targets: list[tuple[str, str | None, str, str, str | None]] = []
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
            revision = rev_candidates[0] if rev_candidates else None
            targets.append((model_id, revision, label, base, target_mxq_path))
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
            targets.append((model_id, revision, label, base, args.mxq_path))

    core_modes = [None] if disable_npu_specific_args else _iter_core_modes_common(args.core_mode)
    run_targets: list[tuple[str, str | None, str, str, str | None, str | None]] = []
    for model_id, revision, label, base, target_mxq_path in targets:
        for core_mode in core_modes:
            mode_label, mode_base = _append_core_mode_suffix_common(label, base, core_mode)
            run_targets.append((model_id, revision, mode_label, mode_base, target_mxq_path, core_mode))

    for model_id, revision, label, base, target_mxq_path, core_mode in tqdm(
        run_targets,
        desc="Benchmarking VLM models",
        unit="model-mode",
    ):
        if _is_cuda_device(args.device):
            _clear_cuda_memory(args.device)
        json_path = results_dir / f"{base}.json"
        csv_path = results_dir / f"{base}.csv"
        png_path = results_dir / f"{base}.png"
        if args.skip_existing and json_path.is_file() and csv_path.is_file() and png_path.is_file():
            print(f"Skipping {label} (results exist).")
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
            payload, rows = _run_model(args, label, pipeline)
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
) -> tuple[Path, bool, list[tuple[str, str | None, str, str, str | None, str | None]]]:
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
    targets: list[tuple[str, str | None, str, str, str | None]] = []
    if args.mxq_dir:
        mxq_dir = Path(args.mxq_dir).expanduser().resolve()
        if not mxq_dir.is_dir():
            raise SystemExit(f"--mxq-dir is not a directory: {mxq_dir}")
        for model_id, rev_candidates, label, base, target_mxq_path in _iter_targets_from_mxq_dir(
            mxq_dir=mxq_dir,
            available_model_ids=available_model_ids,
        ):
            targets.append((model_id, rev_candidates[0] if rev_candidates else None, label, base, target_mxq_path))
    else:
        model_ids = [str(args.model)] if args.model else available_model_ids
        if args.original_models:
            model_ids = _resolve_original_model_ids(model_ids)
        for model_id, revision, label, base in _iter_targets(model_ids, args.revision, args.all):
            targets.append((model_id, revision, label, base, args.mxq_path))
    disable_npu_specific_args = bool(args.original_models and not args.mxq_dir)
    core_modes = [None] if disable_npu_specific_args else _iter_core_modes_common(args.core_mode)
    run_targets: list[tuple[str, str | None, str, str, str | None, str | None]] = []
    for model_id, revision, label, base, target_mxq_path in targets:
        for core_mode in core_modes:
            mode_label, mode_base = _append_core_mode_suffix_common(label, base, core_mode)
            run_targets.append((model_id, revision, mode_label, mode_base, target_mxq_path, core_mode))
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
                "total_energy_j": device.get("total_energy_j"),
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
        return
    rows = _collect_measure_rows(payloads)
    _write_csv(results_dir / "combined_measure.csv", rows)
    _write_measure_markdown(results_dir / "combined_measure.md", rows)
    models = [str(row["model"]) for row in rows]
    chart_specs = [
        ("measure_vision_encode_ms.png", "vision_encode_ms_mean", "ms", "Vision Encode Latency"),
        ("measure_vision_fps.png", "vision_fps_mean", "fps", "Vision FPS"),
        ("measure_llm_prefill_tps.png", "llm_prefill_tps_mean", "tok/s", "LLM Prefill TPS"),
        ("measure_llm_decode_tps.png", "llm_decode_tps_mean", "tok/s", "LLM Decode TPS"),
        ("measure_llm_ttft_ms.png", "llm_ttft_ms_mean", "ms", "LLM TTFT"),
    ]
    for filename, key, x_label, title in chart_specs:
        plot_simple_barh(
            labels=models,
            values=[float(row.get(key) or 0.0) for row in rows],
            x_label=x_label,
            title=title,
            output_path=results_dir / filename,
        )


def _run_measure(args: argparse.Namespace) -> int:
    """Run multi-model image-text-to-text fixed image/prefill/decode benchmarks."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    results_dir, disable_npu_specific_args, run_targets = _collect_vlm_run_targets(args)
    if not run_targets:
        return 0
    if args.rebuild_charts:
        _rebuild_measure_outputs(results_dir)
        return 0
    for model_id, revision, label, base, target_mxq_path, core_mode in tqdm(
        run_targets, desc="Measuring VLM models", unit="model-mode"
    ):
        if _is_cuda_device(args.device):
            _clear_cuda_memory(args.device)
        json_path = results_dir / f"{base}_measure.json"
        if args.skip_existing and json_path.is_file():
            print(f"Skipping {label} (measure result exists).")
            continue
        pipeline = None
        try:
            pipeline = _build_pipeline(args, model_id, revision, target_mxq_path, core_mode)
            measurer = VLMTPSMeasurer(pipeline)
            resolved_prefill_chunk_size = None if disable_npu_specific_args else args.prefill_chunk_size
            for _ in tqdm(range(args.warmup), desc=f"{label} warmup", leave=False):
                measurer.measure_vision(args.image_resolution, repeat=1, prompt=args.prompt, show_progress=False)
                measurer.measure_llm_full(
                    image_resolution=args.image_resolution,
                    prompt=args.prompt,
                    prefill_range=(args.prefill, args.prefill, args.prefill),
                    cache_lengths=[args.prefill],
                    decode_window=args.decode,
                    prefill_chunk_size=resolved_prefill_chunk_size,
                    show_progress=False,
                )
            vision_runs: list[dict[str, float]] = []
            llm_runs: list[dict[str, Any]] = []
            for _ in tqdm(range(args.repeat), desc=f"{label} measured runs", leave=False):
                vision_latency, vision_fps = measurer.measure_vision(
                    args.image_resolution, repeat=1, prompt=args.prompt, show_progress=False
                )[0]
                llm_result = measurer.measure_llm_full(
                    image_resolution=args.image_resolution,
                    prompt=args.prompt,
                    prefill_range=(args.prefill, args.prefill, args.prefill),
                    cache_lengths=[args.prefill],
                    decode_window=args.decode,
                    prefill_chunk_size=resolved_prefill_chunk_size,
                    show_progress=False,
                )
                vision_runs.append({"vision_encode_latency": vision_latency, "vision_fps": vision_fps})
                llm_runs.append(asdict(llm_result))
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
            payload = {
                "model": label,
                "benchmark_type": "measure",
                "task": "image-text-to-text",
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
                "device": None,
                "device_time_series_runs": [],
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
