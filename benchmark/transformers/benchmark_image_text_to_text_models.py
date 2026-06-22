from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chart_utils import ModelMetrics, plot_scalar_chart, plot_token_chart
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
from benchmark.common.summary_utils import read_csv_rows as _read_csv_rows_common
from benchmark.common.summary_utils import scalar_plot_table as _scalar_plot_table_common
from benchmark.common.summary_utils import token_sweep_plot_table as _token_sweep_plot_table_common
from benchmark.common.summary_utils import write_summary_markdown as _write_summary_markdown
from benchmark.common.summary_utils import write_token_combined_markdown as _write_token_combined_markdown
from benchmark.transformers.benchmark_target_utils import (
    args_for_target_device_backend as _args_for_target_device_backend_shared,
)
from benchmark.transformers.benchmark_target_utils import iter_revision_targets as _iter_revision_targets_shared
from benchmark.transformers.benchmark_target_utils import (
    resolve_original_model_ids as _resolve_original_model_ids_shared,
)
from benchmark.transformers.benchmark_target_utils import revision_exists as _revision_exists_shared
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
    build_phase_trackers as _build_phase_trackers,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    energy_from_device_time_series as _energy_from_device_time_series,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    extract_device_metric as _extract_device_metric,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    extract_device_time_series as _extract_device_time_series,
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
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    weighted_two as _weighted_two,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
    BenchmarkResult,
    SweepData,
    VLMTPSMeasurer,
    npu_latency_pct,
)

_VLM_WARMUP_PREFILL = 128
_VLM_WARMUP_DECODE = 32

try:
    from benchmark_text_generation_models import (
        _cuda_memory_info,
        _estimate_model_weight_bytes,
        _filter_text_targets_by_batch_mode,
        _format_gib,
        _is_cuda_oom_error,
        _iter_core_modes_for_target,
        _iter_targets_from_mxq_dir,
        _read_raw_config,
        _should_precheck_cuda,
        _target_sweep_lengths,
    )
except ImportError:
    from .benchmark_text_generation_models import (
        _cuda_memory_info,
        _estimate_model_weight_bytes,
        _filter_text_targets_by_batch_mode,
        _format_gib,
        _is_cuda_oom_error,
        _iter_core_modes_for_target,
        _iter_targets_from_mxq_dir,
        _read_raw_config,
        _should_precheck_cuda,
        _target_sweep_lengths,
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
    batch_mode: str


def _safe_filename(text: str) -> str:
    return _safe_filename_common(text, replace_slash_only=True)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sum_required_energies(*energies: float | None) -> float | None:
    """Return total energy only when every required phase was measured."""
    if any(energy is None for energy in energies):
        return None
    return sum(float(energy) for energy in energies if energy is not None)


def _vlm_llm_run_payload(run: Any) -> dict[str, Any]:
    """Return a serialized VLM LLM run with dynamic phase energy fields."""
    payload = asdict(run)
    for key in (
        "llm_prefill_energy_j",
        "llm_decode_energy_j",
        "llm_total_energy_j",
        "vision_energy_j",
        "total_energy_j",
        "prefill_tps_per_w",
        "decode_tps_per_w",
        "prefill_j_per_token",
        "decode_j_per_token",
        "total_tps_per_w",
        "total_j_per_token",
    ):
        payload[key] = getattr(run, key, None)
    payload["llm_prefill_energy_j"] = getattr(run, "llm_prefill_energy_j", getattr(run, "prefill_energy_j", None))
    payload["llm_decode_energy_j"] = getattr(run, "llm_decode_energy_j", getattr(run, "decode_energy_j", None))
    return payload



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


def _summary(values: Sequence[float]) -> dict[str, float]:
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


def _sweep_prefill_token_count(result: BenchmarkResult, batch_size: int) -> int:
    """Return the number of prefill tokens covered by a VLM LLM sweep."""
    return sum(int(value) for value in result.prefill_sweep.x_values) * max(1, int(batch_size))


def _sweep_decode_token_count(result: BenchmarkResult, *, decode_window: int, batch_size: int) -> int:
    """Return the number of decode tokens covered by a VLM LLM sweep."""
    return int(decode_window) * len(result.decode_sweep.x_values) * max(1, int(batch_size))


def _vlm_warmup_llm_kwargs() -> dict[str, Any]:
    """Return the lightweight VLM LLM warmup dimensions."""
    return {
        "prefill_range": (_VLM_WARMUP_PREFILL, _VLM_WARMUP_PREFILL, _VLM_WARMUP_PREFILL),
        "cache_lengths": [_VLM_WARMUP_PREFILL],
        "decode_window": _VLM_WARMUP_DECODE,
    }


def _build_pipeline(
    args: argparse.Namespace,
    model_id: str,
    revision: str | None,
    mxq_path: str | None,
    core_mode: str | None,
    default_single_target_cores: Sequence[str] | None = ("0:0",),
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
    model_kwargs = _apply_vlm_core_mode_model_kwargs(
        model_kwargs,
        core_mode,
        default_single_target_cores=default_single_target_cores,
    )
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


def _apply_vlm_core_mode_model_kwargs(
    model_kwargs: dict[str, Any],
    core_mode: str | None,
    *,
    default_single_target_cores: Sequence[str] | None = ("0:0",),
) -> dict[str, Any]:
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
    _apply_core_mode_model_kwargs_common(
        expanded,
        core_mode,
        default_single_target_cores=default_single_target_cores,
    )

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
        path for path in mxq_paths if not any(path in files or os.path.basename(path) in files for files in file_sets)
    ]
    if missing:
        return False, f"referenced MXQ artifact(s) not found: {', '.join(missing)}"
    return True, None


def _iter_targets(
    model_ids: list[str], revision: str | None, all_revisions: bool
) -> list[tuple[str, str | None, str, str]]:
    return [
        (model_id, revision_candidates[0], label, base)
        for model_id, revision_candidates, label, base, _mxq_path in _iter_revision_targets_shared(
            model_ids,
            revision=revision,
            all_revisions=all_revisions,
            safe_filename=_safe_filename,
        )
    ]


def _resolve_original_model_ids(model_ids: list[str]) -> list[str]:
    """Resolve Mobilint model ids to parent/original Hugging Face model ids."""
    return _resolve_original_model_ids_shared(model_ids)


def _revision_exists(model_id: str, revision: str) -> bool | None:
    """Check whether a Hugging Face model revision exists."""
    return _revision_exists_shared(model_id, revision)


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

    llm_resolution = args.llm_resolution if args.llm_resolution is not None else args.image_resolutions[0]
    resolved_prefill_chunk_size = None if args.original_models and not args.mxq_dir else args.prefill_chunk_size
    warmup_resolution = llm_resolution
    warmup_llm_kwargs = _vlm_warmup_llm_kwargs()
    for warmup_idx in tqdm(
        range(args.warmup),
        desc=f"{label} warmup@{warmup_resolution}",
        leave=False,
    ):
        measurer.measure_vision(
            image_resolution=warmup_resolution,
            repeat=1,
            prompt=args.prompt,
            batch_size=args.batch_size,
            show_progress=False,
        )
        measurer.measure_llm_full(
            image_resolution=warmup_resolution,
            prompt=args.prompt,
            **warmup_llm_kwargs,
            prefill_chunk_size=resolved_prefill_chunk_size,
            batch_size=args.batch_size,
            show_progress=True,
            progress_prefix=f"{label} warmup {warmup_idx + 1}/{args.warmup}",
        )

    for resolution in args.image_resolutions:
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
                device_time_series = _extract_device_time_series(tracker)
                device_time_series_runs.append(device_time_series)
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
                energy = _energy_from_device_time_series(device_time_series)
                if energy is not None:
                    energy_vals.append(energy)
                    j_per_img = _safe_div(energy, float(args.batch_size))
                    if j_per_img is not None:
                        j_per_img_vals.append(j_per_img)
                    tpj = _safe_div(float(args.batch_size), energy)
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
                    "prefill_tps_per_w": None,
                    "decode_tps_per_w": None,
                    "prefill_j_per_tok": None,
                    "decode_j_per_tok": None,
                    "vision_img_per_j": img_per_j_vals[idx - 1] if idx - 1 < len(img_per_j_vals) else None,
                    "vision_j_per_img": j_per_img_vals[idx - 1] if idx - 1 < len(j_per_img_vals) else None,
                }
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
    llm_prefill_energy_j: list[float] = []
    llm_decode_energy_j: list[float] = []
    llm_total_energy_j: list[float] = []
    vlm_total_energy_j: list[float] = []
    llm_prefill_tps_per_w: list[float] = []
    llm_decode_tps_per_w: list[float] = []
    llm_prefill_j_per_tok: list[float] = []
    llm_decode_j_per_tok: list[float] = []
    llm_device_time_series_runs: list[dict[str, dict[str, list[dict[str, float]]]]] = []
    for repeat_idx in tqdm(
        range(args.repeat),
        desc=f"{label} llm@{llm_resolution} runs",
        leave=False,
    ):
        llm_tracker_prefill, llm_tracker_decode = _build_phase_trackers(args, pipeline)
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
                on_prefill_start=(lambda: llm_tracker_prefill.start()) if llm_tracker_prefill is not None else None,
                on_prefill_end=(lambda: llm_tracker_prefill.stop()) if llm_tracker_prefill is not None else None,
                on_decode_start=(lambda: llm_tracker_decode.start()) if llm_tracker_decode is not None else None,
                on_decode_end=(lambda: llm_tracker_decode.stop()) if llm_tracker_decode is not None else None,
            )
        finally:
            _stop_tracker_safe(llm_tracker_prefill)
            _stop_tracker_safe(llm_tracker_decode)
        if llm_tracker_prefill is not None and llm_tracker_decode is not None:
            prefill_metric = _extract_device_metric(llm_tracker_prefill)
            decode_metric = _extract_device_metric(llm_tracker_decode)
            prefill_time_series = _extract_device_time_series(llm_tracker_prefill)
            decode_time_series = _extract_device_time_series(llm_tracker_decode)
            llm_device_time_series_runs.append({"prefill": prefill_time_series, "decode": decode_time_series})
            prefill_duration = float(getattr(run, "prefill_phase_duration_s", 0.0) or 0.0)
            decode_duration = float(getattr(run, "decode_phase_duration_s", 0.0) or 0.0)
            run.avg_power_w = _weighted_two(
                prefill_metric.get("avg_power_w"),
                prefill_duration,
                decode_metric.get("avg_power_w"),
                decode_duration,
            )
            run.p99_power_w = max(
                [v for v in (prefill_metric.get("p99_power_w"), decode_metric.get("p99_power_w")) if v is not None],
                default=None,
            )
            run.avg_utilization_pct = _weighted_two(
                prefill_metric.get("avg_utilization_pct"),
                prefill_duration,
                decode_metric.get("avg_utilization_pct"),
                decode_duration,
            )
            run.p99_utilization_pct = max(
                [
                    v
                    for v in (prefill_metric.get("p99_utilization_pct"), decode_metric.get("p99_utilization_pct"))
                    if v is not None
                ],
                default=None,
            )
            run.avg_temperature_c = _weighted_two(
                prefill_metric.get("avg_temperature_c"),
                prefill_duration,
                decode_metric.get("avg_temperature_c"),
                decode_duration,
            )
            run.p99_temperature_c = max(
                [
                    v
                    for v in (prefill_metric.get("p99_temperature_c"), decode_metric.get("p99_temperature_c"))
                    if v is not None
                ],
                default=None,
            )
            run.avg_memory_used_mb = _weighted_two(
                prefill_metric.get("avg_memory_used_mb"),
                prefill_duration,
                decode_metric.get("avg_memory_used_mb"),
                decode_duration,
            )
            run.p99_memory_used_mb = max(
                [
                    v
                    for v in (prefill_metric.get("p99_memory_used_mb"), decode_metric.get("p99_memory_used_mb"))
                    if v is not None
                ],
                default=None,
            )
            run.avg_memory_used_pct = _weighted_two(
                prefill_metric.get("avg_memory_used_pct"),
                prefill_duration,
                decode_metric.get("avg_memory_used_pct"),
                decode_duration,
            )
            run.p99_memory_used_pct = max(
                [
                    v
                    for v in (prefill_metric.get("p99_memory_used_pct"), decode_metric.get("p99_memory_used_pct"))
                    if v is not None
                ],
                default=None,
            )
            prefill_energy = _energy_from_device_time_series(prefill_time_series)
            decode_energy = _energy_from_device_time_series(decode_time_series)
            total_energy = _sum_required_energies(prefill_energy, decode_energy)
            run.llm_prefill_energy_j = prefill_energy
            run.llm_decode_energy_j = decode_energy
            run.llm_total_energy_j = total_energy
            run.total_energy_j = total_energy
            if prefill_energy is not None:
                llm_prefill_energy_j.append(prefill_energy)
            if decode_energy is not None:
                llm_decode_energy_j.append(decode_energy)
            if total_energy is not None:
                llm_total_energy_j.append(total_energy)
                prefill_tokens = _sweep_prefill_token_count(run, args.batch_size)
                decode_tokens = _sweep_decode_token_count(
                    run,
                    decode_window=args.decode_window,
                    batch_size=args.batch_size,
                )
                total_tokens = prefill_tokens + decode_tokens
                run.prefill_tps_per_w = _safe_div(float(prefill_tokens), prefill_energy)
                run.decode_tps_per_w = _safe_div(float(decode_tokens), decode_energy)
                run.prefill_j_per_token = (
                    _safe_div(prefill_energy, float(prefill_tokens)) if prefill_tokens > 0 else None
                )
                run.decode_j_per_token = _safe_div(decode_energy, float(decode_tokens)) if decode_tokens > 0 else None
                run.total_tps_per_w = _safe_div(float(total_tokens), total_energy)
                run.total_j_per_token = _safe_div(total_energy, float(total_tokens)) if total_tokens > 0 else None
        llm_runs.append(run)
    llm_prefill = [float(r.prefill_sweep.tps_values[-1]) for r in llm_runs if r.prefill_sweep.tps_values]
    llm_decode = [float(r.decode_sweep.tps_values[-1]) for r in llm_runs if r.decode_sweep.tps_values]
    llm_ttft_values_ms = [
        float(r.prefill_sweep.time_values[-1] * 1000.0) for r in llm_runs if r.prefill_sweep.time_values
    ]
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
    reference_vision_energy_j = None
    for row in vision_results:
        if row.get("image_resolution") != llm_resolution:
            continue
        value = row.get("summary", {}).get("total_energy_j", {}).get("mean")
        if isinstance(value, (int, float)):
            reference_vision_energy_j = float(value)
        break
    for run in llm_runs:
        llm_only_energy = getattr(run, "llm_total_energy_j", getattr(run, "total_energy_j", None))
        run.llm_total_energy_j = llm_only_energy
        run.vision_energy_j = reference_vision_energy_j
        run.total_energy_j = _sum_required_energies(reference_vision_energy_j, llm_only_energy)
    llm_prefill_energy_j = [
        float(r.llm_prefill_energy_j) for r in llm_runs if getattr(r, "llm_prefill_energy_j", None) is not None
    ]
    llm_decode_energy_j = [
        float(r.llm_decode_energy_j) for r in llm_runs if getattr(r, "llm_decode_energy_j", None) is not None
    ]
    llm_total_energy_j = [
        float(r.llm_total_energy_j) for r in llm_runs if getattr(r, "llm_total_energy_j", None) is not None
    ]
    vlm_total_energy_j = [float(r.total_energy_j) for r in llm_runs if getattr(r, "total_energy_j", None) is not None]
    llm_prefill_tps_per_w = [
        float(r.prefill_tps_per_w) for r in llm_runs if getattr(r, "prefill_tps_per_w", None) is not None
    ]
    llm_decode_tps_per_w = [
        float(r.decode_tps_per_w) for r in llm_runs if getattr(r, "decode_tps_per_w", None) is not None
    ]
    llm_prefill_j_per_tok = [
        float(r.prefill_j_per_token) for r in llm_runs if getattr(r, "prefill_j_per_token", None) is not None
    ]
    llm_decode_j_per_tok = [
        float(r.decode_j_per_token) for r in llm_runs if getattr(r, "decode_j_per_token", None) is not None
    ]
    for idx, r in enumerate(llm_runs, start=1):
        for row in BenchmarkResult.iter_rows(label, r):
            phase = str(row.get("phase"))
            csv_rows.append(
                {
                    "type": "llm",
                    "image_resolution": llm_resolution,
                    "repeat_index": idx,
                    "model": row.get("model"),
                    "phase": phase,
                    "tokens": row.get("tokens"),
                    "tps": row.get("tps"),
                    "time_ms": row.get("time_ms"),
                    "avg_total_token_latency_ms": row.get("avg_total_token_latency_ms"),
                    "avg_npu_token_latency_ms": row.get("avg_npu_token_latency_ms"),
                    "avg_npu_token_latency_pct": row.get("avg_npu_token_latency_pct"),
                    "decode_prefill_mode": row.get("decode_prefill_mode"),
                    "vision_encode_ms": None,
                    "vision_fps": None,
                    "llm_prefill_tps": row.get("tps") if phase == "prefill" else None,
                    "llm_decode_tps": row.get("tps") if phase == "decode" else None,
                    "llm_ttft_ms": row.get("time_ms") if phase == "prefill" else None,
                    "llm_decode_duration_ms": row.get("time_ms") if phase == "decode" else None,
                    "llm_prefill_npu_latency_pct": row.get("avg_npu_token_latency_pct") if phase == "prefill" else None,
                    "llm_decode_npu_latency_pct": row.get("avg_npu_token_latency_pct") if phase == "decode" else None,
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
                    "vision_energy_j": getattr(r, "vision_energy_j", None),
                    "llm_prefill_energy_j": getattr(r, "llm_prefill_energy_j", None),
                    "llm_decode_energy_j": getattr(r, "llm_decode_energy_j", None),
                    "llm_total_energy_j": getattr(r, "llm_total_energy_j", None),
                    "total_energy_j": getattr(r, "total_energy_j", None),
                    "prefill_tps_per_w": getattr(r, "prefill_tps_per_w", None),
                    "decode_tps_per_w": getattr(r, "decode_tps_per_w", None),
                    "prefill_j_per_tok": getattr(r, "prefill_j_per_token", None),
                    "decode_j_per_tok": getattr(r, "decode_j_per_token", None),
                    "vision_img_per_j": None,
                    "vision_j_per_img": None,
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
                "runs": [_vlm_llm_run_payload(r) for r in llm_runs],
                "device_time_series_runs": llm_device_time_series_runs,
                "summary": {
                    "llm_prefill_tps": _summary(llm_prefill),
                    "llm_decode_tps": _summary(llm_decode),
                    "llm_ttft_ms": _summary(llm_ttft_values_ms),
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
                    "llm_prefill_energy_j": _summary(llm_prefill_energy_j),
                    "llm_decode_energy_j": _summary(llm_decode_energy_j),
                    "llm_total_energy_j": _summary(llm_total_energy_j),
                    "vision_energy_j": _summary(
                        [reference_vision_energy_j] if reference_vision_energy_j is not None else []
                    ),
                    "total_energy_j": _summary(vlm_total_energy_j),
                    "prefill_tps_per_w": _summary(llm_prefill_tps_per_w),
                    "decode_tps_per_w": _summary(llm_decode_tps_per_w),
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
            "vision_energy_j": reference_vision_energy_j,
            "llm_prefill_energy_j": _mean(llm_prefill_energy_j) if llm_prefill_energy_j else None,
            "llm_decode_energy_j": _mean(llm_decode_energy_j) if llm_decode_energy_j else None,
            "llm_total_energy_j": _mean(llm_total_energy_j) if llm_total_energy_j else None,
            "total_energy_j": _mean(vlm_total_energy_j) if vlm_total_energy_j else None,
            "prefill_tps_last": llm_prefill[-1] if llm_prefill else None,
            "decode_tps_last": llm_decode[-1] if llm_decode else None,
            "prefill_tps_per_w_last": llm_prefill_tps_per_w[-1] if llm_prefill_tps_per_w else None,
            "decode_tps_per_w_last": llm_decode_tps_per_w[-1] if llm_decode_tps_per_w else None,
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


def _sweep_from_dict(payload: Mapping[str, Any]) -> SweepData:
    """Build ``SweepData`` from a serialized sweep dictionary."""
    x_values = list(payload.get("x_values", []))

    def _latency_values(key: str) -> list[Any]:
        values = list(payload.get(key, []))
        return values + [None] * max(0, len(x_values) - len(values))

    return SweepData(
        x_values=x_values,
        tps_values=list(payload.get("tps_values", [])),
        time_values=list(payload.get("time_values", [])),
        avg_total_token_latency_values=_latency_values("avg_total_token_latency_values"),
        avg_npu_token_latency_values=_latency_values("avg_npu_token_latency_values"),
    )


def _benchmark_result_from_vlm_run(payload: Mapping[str, Any]) -> BenchmarkResult:
    """Build ``BenchmarkResult`` from one serialized VLM LLM run."""
    return BenchmarkResult(
        prefill_sweep=_sweep_from_dict(payload.get("prefill_sweep", {})),
        decode_sweep=_sweep_from_dict(payload.get("decode_sweep", {})),
        decode_prefill_modes=list(payload.get("decode_prefill_modes", [])),
        prefill_phase_duration_s=payload.get("prefill_phase_duration_s"),
        decode_phase_duration_s=payload.get("decode_phase_duration_s"),
        avg_power_w=payload.get("avg_power_w"),
        p99_power_w=payload.get("p99_power_w"),
        avg_utilization_pct=payload.get("avg_utilization_pct"),
        p99_utilization_pct=payload.get("p99_utilization_pct"),
        avg_temperature_c=payload.get("avg_temperature_c"),
        p99_temperature_c=payload.get("p99_temperature_c"),
        avg_memory_used_mb=payload.get("avg_memory_used_mb"),
        p99_memory_used_mb=payload.get("p99_memory_used_mb"),
        avg_memory_used_pct=payload.get("avg_memory_used_pct"),
        p99_memory_used_pct=payload.get("p99_memory_used_pct"),
        total_energy_j=payload.get("total_energy_j"),
        prefill_tps_per_w=payload.get("prefill_tps_per_w"),
        prefill_j_per_token=payload.get("prefill_j_per_token"),
        decode_tps_per_w=payload.get("decode_tps_per_w"),
        decode_j_per_token=payload.get("decode_j_per_token"),
    )


def _mean_optional(values: Sequence[Any]) -> float | None:
    """Return a mean over numeric values, or ``None`` when no numeric values exist."""
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def _aggregate_vlm_llm_runs(runs: Sequence[Mapping[str, Any]]) -> BenchmarkResult | None:
    """Aggregate serialized VLM LLM runs into the same shape used by LLM combined outputs."""
    results = [_benchmark_result_from_vlm_run(run) for run in runs]
    if not results:
        return None
    if len(results) == 1:
        return results[0]

    def _aggregate_phase(phase: str) -> SweepData:
        first = results[0].prefill_sweep if phase == "prefill" else results[0].decode_sweep
        out = SweepData(x_values=list(first.x_values))

        def _get_optional(values: Sequence[Any], idx: int) -> Any:
            return values[idx] if idx < len(values) else None

        for idx in range(len(first.x_values)):
            tps_values: list[float] = []
            time_values: list[float] = []
            total_latency_values: list[float | None] = []
            npu_latency_values: list[float | None] = []
            for result in results:
                src = result.prefill_sweep if phase == "prefill" else result.decode_sweep
                tps_values.append(float(src.tps_values[idx]))
                time_values.append(float(src.time_values[idx]))
                total_latency_values.append(_get_optional(src.avg_total_token_latency_values, idx))
                npu_latency_values.append(_get_optional(src.avg_npu_token_latency_values, idx))
            out.tps_values.append(sum(tps_values) / len(tps_values))
            out.time_values.append(sum(time_values) / len(time_values))
            out.avg_total_token_latency_values.append(_mean_optional(total_latency_values))
            out.avg_npu_token_latency_values.append(_mean_optional(npu_latency_values))
        return out

    return BenchmarkResult(
        prefill_sweep=_aggregate_phase("prefill"),
        decode_sweep=_aggregate_phase("decode"),
        decode_prefill_modes=list(results[0].decode_prefill_modes),
    )


def _vlm_metrics_from_result(payload: Mapping[str, Any], result: BenchmarkResult) -> ModelMetrics:
    """Convert one VLM aggregate result into chart ``ModelMetrics``."""
    raw_device = payload.get("device")
    device: Mapping[str, Any] = raw_device if isinstance(raw_device, Mapping) else {}

    def _token_map(sweep: SweepData, values: Sequence[Any]) -> dict[int, float]:
        return {
            int(token): float(value)
            for token, value in zip(sweep.x_values, values)
            if isinstance(token, int) and isinstance(value, (int, float))
        }

    return ModelMetrics(
        prefill_tps=_token_map(result.prefill_sweep, result.prefill_sweep.tps_values),
        decode_tps=_token_map(result.decode_sweep, result.decode_sweep.tps_values),
        prefill_latency_ms=_token_map(
            result.prefill_sweep,
            [value * 1000.0 for value in result.prefill_sweep.time_values],
        ),
        decode_duration_ms=_token_map(
            result.decode_sweep,
            [value * 1000.0 for value in result.decode_sweep.time_values],
        ),
        prefill_tps_per_w=_as_float(device.get("prefill_tps_per_w_last")),
        decode_tps_per_w=_as_float(device.get("decode_tps_per_w_last")),
        prefill_j_per_token=_as_float(device.get("prefill_j_per_tok_last")),
        decode_j_per_token=_as_float(device.get("decode_j_per_tok_last")),
        avg_power_w=_as_float(device.get("avg_power_w")),
        total_energy_j=_as_float(device.get("total_energy_j")),
        avg_utilization_pct=_as_float(device.get("avg_utilization_pct")),
        p99_utilization_pct=_as_float(device.get("p99_utilization_pct")),
        avg_temperature_c=_as_float(device.get("avg_temperature_c")),
        p99_temperature_c=_as_float(device.get("p99_temperature_c")),
        avg_memory_used_mb=_as_float(device.get("avg_memory_used_mb")),
        p99_memory_used_mb=_as_float(device.get("p99_memory_used_mb")),
        avg_memory_used_pct=_as_float(device.get("avg_memory_used_pct")),
        p99_memory_used_pct=_as_float(device.get("p99_memory_used_pct")),
    )


def _as_float(value: Any) -> float | None:
    """Return a float for numeric values."""
    return float(value) if isinstance(value, (int, float)) else None


def _rebuild_combined(output_dir: Path) -> None:
    llm_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    vision_rows: list[dict[str, Any]] = []
    metrics_by_model: dict[str, ModelMetrics] = {}
    for path in sorted(output_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("benchmark_type") == "measure" or path.name == _HOST_PC_INFO_FILENAME:
            continue
        if not isinstance(payload.get("model"), str) or not isinstance(payload.get("benchmark"), dict):
            continue
        label = str(payload["model"])
        bench = payload.get("benchmark", {})
        llm_results = bench.get("llm_results", {}) if isinstance(bench.get("llm_results"), dict) else {}
        llm_runs = llm_results.get("runs", []) if isinstance(llm_results.get("runs"), list) else []
        result = _aggregate_vlm_llm_runs([run for run in llm_runs if isinstance(run, dict)])
        if result is not None:
            llm_rows.extend(list(BenchmarkResult.iter_rows(label, result)))
            metrics_by_model[label] = _vlm_metrics_from_result(payload, result)
        device = payload.get("device")
        if isinstance(device, dict):
            device_rows.append({"model": label, **device})
        for row in bench.get("vision_results", []):
            if not isinstance(row, dict):
                continue
            s = row.get("summary", {})
            vision_rows.append(
                {
                    "model": label,
                    "batch_mode": payload.get("batch_mode"),
                    "batch_size": payload.get("batch_size"),
                    "image_resolution": row.get("image_resolution"),
                    "vision_encode_ms_mean": s.get("vision_encode_ms", {}).get("mean"),
                    "vision_fps_mean": s.get("vision_fps", {}).get("mean"),
                    "vision_img_per_j_mean": s.get("vision_img_per_j", {}).get("mean"),
                    "vision_j_per_img_mean": s.get("vision_j_per_img", {}).get("mean"),
                }
            )
    BenchmarkResult.write_combined_csv(str(output_dir / "combined.csv"), llm_rows)
    BenchmarkResult.write_combined_csv(str(output_dir / "combined_llm.csv"), llm_rows)
    _write_csv(output_dir / "combined_vision.csv", vision_rows)
    _write_csv(output_dir / "combined_device.csv", device_rows)
    if not llm_rows:
        _write_vlm_summary(output_dir)
        return
    _write_token_combined_markdown(output_dir / "combined.md", llm_rows, device_rows)
    if metrics_by_model:
        models = sorted(metrics_by_model.keys())
        labels = ["benchmark"]
        metrics_by_folder = [metrics_by_model]
        plot_token_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            token_selector=lambda metric: metric.prefill_tps,
            title="Prefill Tokens Per Second",
            x_label="Tokens Per Second",
            output_path=output_dir / "llm_prefill_tps.png",
        )
        plot_token_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            token_selector=lambda metric: metric.decode_tps,
            title="Decode Tokens Per Second",
            x_label="Tokens Per Second",
            output_path=output_dir / "llm_decode_tps.png",
        )
        scalar_specs = [
            (
                "llm_prefill_tps_per_w.png",
                "Prefill TPS/W",
                "TPS/W",
                lambda m: m.prefill_tps_per_w,
            ),
            (
                "llm_decode_tps_per_w.png",
                "Decode TPS/W",
                "TPS/W",
                lambda m: m.decode_tps_per_w,
            ),
            ("avg_power_w.png", "Power", "Power (Watts)", lambda m: m.avg_power_w),
            ("avg_temperature_c.png", "Temperature", "Temperature (Celsius)", lambda m: m.avg_temperature_c),
            ("avg_utilization_pct.png", "Utilization", "Utilization (Percent)", lambda m: m.avg_utilization_pct),
            (
                "avg_memory_used_mb.png",
                "Memory Used Megabytes",
                "Memory Used (Megabytes)",
                lambda m: m.avg_memory_used_mb,
            ),
            ("total_energy_j.png", "Total Energy", "Energy (Joules)", lambda m: m.total_energy_j),
        ]
        for filename, title, x_label, selector in scalar_specs:
            plot_scalar_chart(
                models=models,
                folder_labels=labels,
                metrics_by_folder=metrics_by_folder,
                scalar_selector=selector,
                title=title,
                x_label=x_label,
                output_path=output_dir / filename,
            )
    _write_vlm_summary(output_dir)


def _write_vlm_summary(output_dir: Path, *, measure: bool = False) -> None:
    """Write an image-text-to-text benchmark summary Markdown with host info, plots, and table."""
    table_name = "combined_measure.md" if measure else "combined.md"
    summary_name = "summary_measure.md" if measure else "summary.md"
    title = "Image Text-to-Text Measure Benchmark Summary" if measure else "Image Text-to-Text Benchmark Summary"
    prefixes = ("measure_",) if measure else None
    plot_tables = _build_vlm_plot_tables(output_dir, measure=measure)
    _write_summary_markdown(
        output_dir / summary_name,
        title=title,
        host_info_path=output_dir / _HOST_PC_INFO_FILENAME,
        table_markdown_path=output_dir / table_name,
        plot_paths=_existing_png_paths(output_dir, prefixes=prefixes),
        plot_tables=plot_tables,
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
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--batch",
        dest="batch_mode",
        action="store_const",
        const="batch",
        default="non_batch",
        help="benchmark only batch-capable model targets",
    )
    batch_group.add_argument(
        "--non-batch",
        dest="batch_mode",
        action="store_const",
        const="non_batch",
        help="benchmark only non-batch model targets (default)",
    )
    parser.add_argument("--model", dest="models", nargs="+", default=None, help="model id list to benchmark (optional)")
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
        "--output-dir",
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


def _default_single_target_cores_for_batch_mode(batch_mode: str) -> Sequence[str] | None:
    """Return the default single-mode target cores for one resolved target batch mode."""
    if isinstance(batch_mode, argparse.Namespace):
        batch_mode = str(getattr(batch_mode, "batch_mode", "non_batch"))
    if batch_mode == "batch":
        return None
    return ("0:0",)


def _resolve_runtime_defaults(args: argparse.Namespace, raw_argv: list[str]) -> None:
    """Apply benchmark runtime defaults that depend on explicit CLI flags."""
    device_explicit = _flag_present(raw_argv, "--device")
    device_backend_explicit = _flag_present(raw_argv, "--device-backend")
    core_mode_explicit = _flag_present(raw_argv, "--core-mode")
    first_model_id = None if args.mxq_dir else ((args.models or [None])[0])
    args._device_explicit = device_explicit
    args._device_requested = args.device
    args._device_backend_explicit = device_backend_explicit
    args._device_backend_requested = args.device_backend
    args._core_mode_explicit = core_mode_explicit
    if args.batch_mode == "batch":
        if core_mode_explicit and args.core_mode != "single":
            raise SystemExit("batch benchmark only supports --core-mode single")
        args.core_mode = "single"
    args.device = _resolve_default_device_common(
        device=args.device,
        device_explicit=device_explicit,
        model_id=first_model_id,
        mxq_path=args.mxq_path,
        mxq_dir=args.mxq_dir,
        original_models=args.original_models,
    )
    if not device_explicit:
        print(f"Auto-set --device={args.device}")
    args.device_backend = _resolve_default_device_backend_common(
        device_backend=args.device_backend,
        device_backend_explicit=device_backend_explicit,
        model_id=first_model_id,
        mxq_path=args.mxq_path,
        mxq_dir=args.mxq_dir,
        original_models=args.original_models,
    )
    if not device_backend_explicit:
        if first_model_id or args.mxq_path or args.mxq_dir:
            print(f"Auto-set --device-backend={args.device_backend} (based on target/device policy)")
        else:
            print("Auto-set --device-backend per target (based on target/device policy)")
    args._raw_argv = list(raw_argv)


def _args_for_target_device_backend(
    args: argparse.Namespace,
    *,
    model_id: str,
    mxq_path: str | None = None,
) -> argparse.Namespace:
    """Return an args copy with a device backend resolved for one benchmark target."""
    return _args_for_target_device_backend_shared(
        args,
        model_id=model_id,
        mxq_path=mxq_path,
        resolve_default_device=_resolve_default_device_common,
        resolve_default_device_backend=_resolve_default_device_backend_common,
    )


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
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else script_dir / "results" / "image_text_to_text"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_charts:
        _rebuild_combined(output_dir)
        return 0

    _collect_host_pc_info(output_dir)

    available_model_ids = list_models(tasks="image-text-to-text").get("image-text-to-text", [])
    if not available_model_ids:
        print("No image-text-to-text models found.")
        return 0
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = []
    if args.mxq_dir:
        mxq_dir = Path(args.mxq_dir).expanduser().resolve()
        if not mxq_dir.is_dir():
            raise SystemExit(f"--mxq-dir is not a directory: {mxq_dir}")
        if args.models or args.original_models or args.all or args.revision:
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
        if args.models:
            model_ids = [str(item) for item in args.models]
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
            batch_mode=target.batch_mode,
        )
        for target in _filter_text_targets_by_batch_mode(
            raw_targets,
            batch_mode=args.batch_mode,
            task="image-text-to-text",
        )
    ]
    run_targets: list[
        tuple[str, str | None, str, str, str | None, str | None, int, str, tuple[int, int, int], list[int]]
    ] = []
    for target in targets:
        target_prefill_range, target_cache_lengths = _target_sweep_lengths(
            args,
            getattr(args, "_raw_argv", []),
            target.batch_mode,
        )
        for core_mode in _iter_core_modes_for_target(
            args,
            target.batch_mode,
            disable_npu_specific_args=disable_npu_specific_args,
        ):
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
                    target.batch_mode,
                    target_prefill_range,
                    target_cache_lengths,
                )
            )

    for (
        model_id,
        revision,
        label,
        base,
        target_mxq_path,
        core_mode,
        batch_size,
        batch_mode,
        prefill_range,
        cache_lengths,
    ) in tqdm(
        run_targets,
        desc="Benchmarking VLM models",
        unit="model-mode",
    ):
        target_args = _args_for_target_device_backend(args, model_id=model_id, mxq_path=target_mxq_path)
        if _is_cuda_device(target_args.device):
            _clear_cuda_memory(target_args.device)
        json_path = output_dir / f"{base}.json"
        csv_path = output_dir / f"{base}.csv"
        png_path = output_dir / f"{base}.png"
        if args.skip_existing and json_path.is_file() and csv_path.is_file() and png_path.is_file():
            print(f"Skipping {label} (results exist).")
            continue
        artifacts_available, skip_reason = _vlm_revision_artifacts_available(model_id, revision, target_mxq_path)
        if not artifacts_available:
            print(f"Skipping {label} ({skip_reason}).")
            continue
        if _should_precheck_cuda(target_args):
            estimated = _estimate_model_weight_bytes(model_id, revision)
            mem_info = _cuda_memory_info(target_args.device)
            if estimated is not None and mem_info is not None:
                free_b, _ = mem_info
                required = int(float(estimated) * float(args.cuda_precheck_margin))
                if free_b < required:
                    print(
                        "Skipping (pre-check VRAM insufficient): "
                        f"free={_format_gib(free_b)} required~={_format_gib(required)} "
                        f"estimated_weights={_format_gib(estimated)}"
                    )
                    _clear_cuda_memory(target_args.device)
                    continue
        pipeline = None
        try:
            pipeline = _build_pipeline(
                target_args,
                model_id,
                revision,
                target_mxq_path,
                core_mode,
                default_single_target_cores=_default_single_target_cores_for_batch_mode(batch_mode),
            )
            target_args.batch_size = batch_size
            target_args.batch_mode = batch_mode
            target_args.prefill_range = prefill_range
            target_args.cache_lengths = cache_lengths
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
            _release_pipeline(pipeline, target_args.device)

    _rebuild_combined(output_dir)
    return 0


def _collect_vlm_run_targets(
    args: argparse.Namespace,
) -> tuple[Path, bool, list[tuple[str, str | None, str, str, str | None, str | None, int, str]]]:
    """Resolve image-text-to-text benchmark targets and core-mode expansion."""
    script_dir = Path(__file__).resolve().parent
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else script_dir / "results" / "image_text_to_text"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    available_model_ids: list[str] | None = None
    if args.mxq_dir or not args.models:
        available_model_ids = list_models(tasks="image-text-to-text").get("image-text-to-text", [])
        if not available_model_ids:
            print("No image-text-to-text models found.")
            return output_dir, False, []
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = []
    if args.mxq_dir:
        mxq_dir = Path(args.mxq_dir).expanduser().resolve()
        if not mxq_dir.is_dir():
            raise SystemExit(f"--mxq-dir is not a directory: {mxq_dir}")
        for model_id, rev_candidates, label, base, target_mxq_path in _iter_targets_from_mxq_dir(
            mxq_dir=mxq_dir,
            available_model_ids=available_model_ids or [],
        ):
            raw_targets.append((model_id, rev_candidates, label, base, target_mxq_path))
    else:
        model_ids = [str(item) for item in args.models] if args.models else (available_model_ids or [])
        if args.original_models:
            model_ids = _resolve_original_model_ids(model_ids)
        for model_id, revision, label, base in _iter_targets(model_ids, args.revision, args.all):
            raw_targets.append((model_id, [revision], label, base, args.mxq_path))
    disable_npu_specific_args = bool(args.original_models and not args.mxq_dir)
    targets = [
        VLMBenchmarkTarget(
            model_id=target.model_id,
            revision=target.revision_candidates[0] if target.revision_candidates else None,
            label=target.label,
            base=target.base,
            mxq_path=target.mxq_path,
            max_batch_size=target.max_batch_size,
            batch_mode=target.batch_mode,
        )
        for target in _filter_text_targets_by_batch_mode(
            raw_targets,
            batch_mode=args.batch_mode,
            task="image-text-to-text",
        )
    ]
    run_targets: list[tuple[str, str | None, str, str, str | None, str | None, int, str]] = []
    for target in targets:
        for core_mode in _iter_core_modes_for_target(
            args,
            target.batch_mode,
            disable_npu_specific_args=disable_npu_specific_args,
        ):
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
                    target.batch_mode,
                )
            )
    return output_dir, disable_npu_specific_args, run_targets


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
                "vision_energy_j": device.get("vision_energy_j"),
                "llm_prefill_energy_j": device.get("llm_prefill_energy_j"),
                "llm_decode_energy_j": device.get("llm_decode_energy_j"),
                "llm_total_energy_j": device.get("llm_total_energy_j"),
                "total_energy_j": device.get("total_energy_j"),
                "llm_prefill_tps_per_w_mean": device.get("llm_prefill_tps_per_w"),
                "llm_decode_tps_per_w_mean": device.get("llm_decode_tps_per_w"),
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


def _escape_markdown_cell(value: str) -> str:
    """Escape text for use inside a Markdown table cell."""
    return value.replace("|", "\\|").replace("\n", "<br>")


def _format_summary_cell(value: Any) -> str:
    """Format one benchmark summary table value."""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    if isinstance(value, str):
        try:
            return f"{float(value):.6f}"
        except ValueError:
            return _escape_markdown_cell(value)
    return _escape_markdown_cell(str(value))


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Build a compact Markdown table with right-aligned metric columns."""
    from benchmark.common.summary_utils import markdown_table as _markdown_table_common

    return _markdown_table_common(headers, rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows if the file exists."""
    return _read_csv_rows_common(path)


def _scalar_plot_table(rows: Sequence[Mapping[str, Any]], *, value_key: str, unit_header: str) -> str:
    """Build a model/value table for one scalar plot."""
    return _scalar_plot_table_common(rows, value_key=value_key, unit_header=unit_header)


def _build_vlm_plot_tables(output_dir: Path, *, measure: bool = False) -> dict[str, str]:
    """Build plot-specific Markdown tables for VLM summaries."""
    rows = _read_csv_rows(output_dir / ("combined_measure.csv" if measure else "combined_device.csv"))
    prefix = "measure_" if measure else ""
    if not measure:
        metrics_by_model = _collect_vlm_combined_metrics(output_dir)
        models = sorted(metrics_by_model.keys())
        tables = {
            "llm_prefill_tps.png": _token_sweep_plot_table_common(models, metrics_by_model, value_key="prefill_tps"),
            "llm_decode_tps.png": _token_sweep_plot_table_common(models, metrics_by_model, value_key="decode_tps"),
        }
        scalar_specs = [
            ("llm_prefill_tps_per_w.png", "prefill_tps_per_w", "TPS/W"),
            ("llm_decode_tps_per_w.png", "decode_tps_per_w", "TPS/W"),
            ("avg_power_w.png", "avg_power_w", "W"),
            ("avg_temperature_c.png", "avg_temperature_c", "°C"),
            ("avg_utilization_pct.png", "avg_utilization_pct", "%"),
            ("avg_memory_used_mb.png", "avg_memory_used_mb", "MB"),
            ("total_energy_j.png", "total_energy_j", "J"),
        ]
        metric_rows = [{"model": model, **vars(metrics_by_model[model])} for model in models]
        for filename, key, unit_header in scalar_specs:
            tables[filename] = _scalar_plot_table(metric_rows, value_key=key, unit_header=unit_header)
        return {filename: table for filename, table in tables.items() if table}

    specs = [
        (f"{prefix}llm_prefill_tps.png", "llm_prefill_tps_mean", "tokens/s"),
        (f"{prefix}llm_prefill_tps_per_w.png", "llm_prefill_tps_per_w_mean", "TPS/W"),
        (f"{prefix}llm_decode_tps.png", "llm_decode_tps_mean", "tokens/s"),
        (f"{prefix}llm_decode_tps_per_w.png", "llm_decode_tps_per_w_mean", "TPS/W"),
        (f"{prefix}avg_power_w.png", "avg_power_w", "W"),
        (f"{prefix}avg_temperature_c.png", "avg_temperature_c", "°C"),
        (f"{prefix}avg_utilization_pct.png", "avg_utilization_pct", "%"),
        (f"{prefix}avg_memory_used_mb.png", "avg_memory_used_mb", "MB"),
        (f"{prefix}vision_energy_j.png", "vision_energy_j", "J"),
        (f"{prefix}llm_prefill_energy_j.png", "llm_prefill_energy_j", "J"),
        (f"{prefix}llm_decode_energy_j.png", "llm_decode_energy_j", "J"),
        (f"{prefix}llm_total_energy_j.png", "llm_total_energy_j", "J"),
        (f"{prefix}total_energy_j.png", "total_energy_j", "J"),
    ]
    if not rows:
        return {}
    return {
        filename: table
        for filename, key, unit_header in specs
        if (table := _scalar_plot_table(rows, value_key=key, unit_header=unit_header))
    }


def _collect_vlm_combined_metrics(output_dir: Path) -> dict[str, ModelMetrics]:
    """Collect VLM token-sweep metrics from JSON result files."""
    metrics: dict[str, ModelMetrics] = {}
    for path in sorted(output_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("benchmark_type") == "measure" or path.name == _HOST_PC_INFO_FILENAME:
            continue
        label = payload.get("model")
        bench = payload.get("benchmark")
        if not isinstance(label, str) or not isinstance(bench, dict):
            continue
        llm_results = bench.get("llm_results") if isinstance(bench.get("llm_results"), dict) else {}
        runs = (
            llm_results.get("runs", [])
            if isinstance(llm_results, dict) and isinstance(llm_results.get("runs"), list)
            else []
        )
        result = _aggregate_vlm_llm_runs([run for run in runs if isinstance(run, dict)])
        if result is not None:
            metrics[label] = _vlm_metrics_from_result(payload, result)
    return metrics


def _rebuild_measure_outputs(output_dir: Path) -> None:
    """Rebuild VLM measure combined outputs."""
    payloads: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*_measure.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("benchmark_type") == "measure":
            payloads.append(payload)
    if not payloads:
        print("No measure JSON results found. Nothing to aggregate.")
        _write_vlm_summary(output_dir, measure=True)
        return
    rows = _collect_measure_rows(payloads)
    _write_csv(output_dir / "combined_measure.csv", rows)
    _write_measure_markdown(output_dir / "combined_measure.md", rows)
    models = [str(row["model"]) for row in rows]
    chart_specs = [
        ("measure_llm_prefill_tps.png", "llm_prefill_tps_mean", "Tokens Per Second", "Prefill Tokens Per Second"),
        (
            "measure_llm_prefill_tps_per_w.png",
            "llm_prefill_tps_per_w_mean",
            "TPS/W",
            "Prefill TPS/W",
        ),
        ("measure_llm_decode_tps.png", "llm_decode_tps_mean", "Tokens Per Second", "Decode Tokens Per Second"),
        (
            "measure_llm_decode_tps_per_w.png",
            "llm_decode_tps_per_w_mean",
            "TPS/W",
            "Decode TPS/W",
        ),
        ("measure_avg_power_w.png", "avg_power_w", "Power (Watts)", "Power"),
        ("measure_avg_temperature_c.png", "avg_temperature_c", "Temperature (Celsius)", "Temperature"),
        ("measure_avg_utilization_pct.png", "avg_utilization_pct", "Utilization (Percent)", "Utilization"),
        ("measure_avg_memory_used_mb.png", "avg_memory_used_mb", "Memory Used (Megabytes)", "Memory Used Megabytes"),
        ("measure_vision_energy_j.png", "vision_energy_j", "Energy (Joules)", "Vision Energy"),
        ("measure_llm_prefill_energy_j.png", "llm_prefill_energy_j", "Energy (Joules)", "LLM Prefill Energy"),
        ("measure_llm_decode_energy_j.png", "llm_decode_energy_j", "Energy (Joules)", "LLM Decode Energy"),
        ("measure_llm_total_energy_j.png", "llm_total_energy_j", "Energy (Joules)", "LLM Total Energy"),
        ("measure_total_energy_j.png", "total_energy_j", "Energy (Joules)", "Total Energy"),
    ]
    for filename, key, x_label, title in chart_specs:
        plot_simple_barh(
            labels=models,
            values=[float(row.get(key) or 0.0) for row in rows],
            x_label=x_label,
            title=title,
            output_path=output_dir / filename,
        )
    _write_vlm_summary(output_dir, measure=True)


def _resolve_vlm_output_dir(args: argparse.Namespace) -> Path:
    """Resolve and create the image-text-to-text benchmark output directory."""
    script_dir = Path(__file__).resolve().parent
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else script_dir / "results" / "image_text_to_text"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _run_measure(args: argparse.Namespace) -> int:
    """Run multi-model image-text-to-text fixed image/prefill/decode benchmarks."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.rebuild_charts:
        _rebuild_measure_outputs(_resolve_vlm_output_dir(args))
        return 0
    output_dir, disable_npu_specific_args, run_targets = _collect_vlm_run_targets(args)
    if not run_targets:
        return 0
    _collect_host_pc_info(output_dir)
    for model_id, revision, label, base, target_mxq_path, core_mode, batch_size, batch_mode in tqdm(
        run_targets, desc="Measuring VLM models", unit="model-mode"
    ):
        target_args = _args_for_target_device_backend(args, model_id=model_id, mxq_path=target_mxq_path)
        target_args.batch_size = batch_size
        target_args.batch_mode = batch_mode
        if _is_cuda_device(target_args.device):
            _clear_cuda_memory(target_args.device)
        json_path = output_dir / f"{base}_measure.json"
        if args.skip_existing and json_path.is_file():
            print(f"Skipping {label} (measure result exists).")
            continue
        artifacts_available, skip_reason = _vlm_revision_artifacts_available(model_id, revision, target_mxq_path)
        if not artifacts_available:
            print(f"Skipping {label} ({skip_reason}).")
            continue
        pipeline = None
        try:
            pipeline = _build_pipeline(
                target_args,
                model_id,
                revision,
                target_mxq_path,
                core_mode,
                default_single_target_cores=_default_single_target_cores_for_batch_mode(batch_mode),
            )
            measurer = VLMTPSMeasurer(pipeline)
            tracker = _build_device_tracker(target_args, pipeline)
            _print_device_status(target_args, tracker)
            resolved_prefill_chunk_size = None if disable_npu_specific_args else args.prefill_chunk_size
            warmup_llm_kwargs = _vlm_warmup_llm_kwargs()
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
                    **warmup_llm_kwargs,
                    prefill_chunk_size=resolved_prefill_chunk_size,
                    batch_size=batch_size,
                    show_progress=True,
                    progress_prefix=f"{label} warmup {warmup_idx + 1}/{args.warmup}",
                )
            vision_runs: list[dict[str, float]] = []
            llm_runs: list[dict[str, Any]] = []
            device_time_series_runs: list[dict[str, Any]] = []
            avg_power_w: list[float] = []
            avg_temperature_c: list[float] = []
            avg_utilization_pct: list[float] = []
            avg_memory_used_mb: list[float] = []
            vision_energy_j: list[float] = []
            llm_prefill_energy_j: list[float] = []
            llm_decode_energy_j: list[float] = []
            llm_total_energy_j: list[float] = []
            vlm_total_energy_j: list[float] = []
            llm_prefill_tps_per_w: list[float] = []
            llm_decode_tps_per_w: list[float] = []
            for repeat_idx in tqdm(range(args.repeat), desc=f"{label} measured runs", leave=False):
                current_vision_energy: float | None = None
                if tracker is not None:
                    tracker.start()
                try:
                    vision_latency_per_image, vision_fps = measurer.measure_vision(
                        args.image_resolution,
                        repeat=1,
                        prompt=args.prompt,
                        batch_size=batch_size,
                        show_progress=False,
                    )[0]
                finally:
                    _stop_tracker_safe(tracker)

                vision_device_time_series: dict[str, list[dict[str, float]]] = {}
                if tracker is not None:
                    metric = _extract_device_metric(tracker)
                    vision_device_time_series = _extract_device_time_series(tracker)
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
                        avg_power_w.append(float(power))
                    energy = _energy_from_device_time_series(vision_device_time_series)
                    if energy is not None:
                        current_vision_energy = energy
                        vision_energy_j.append(energy)

                llm_tracker_prefill, llm_tracker_decode = _build_phase_trackers(target_args, pipeline)
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
                        on_prefill_start=(
                            (lambda: llm_tracker_prefill.start()) if llm_tracker_prefill is not None else None
                        ),
                        on_prefill_end=(
                            (lambda: llm_tracker_prefill.stop()) if llm_tracker_prefill is not None else None
                        ),
                        on_decode_start=(
                            (lambda: llm_tracker_decode.start()) if llm_tracker_decode is not None else None
                        ),
                        on_decode_end=(
                            (lambda: llm_tracker_decode.stop()) if llm_tracker_decode is not None else None
                        ),
                    )
                finally:
                    _stop_tracker_safe(llm_tracker_prefill)
                    _stop_tracker_safe(llm_tracker_decode)

                llm_device_time_series: dict[str, dict[str, list[dict[str, float]]]] = {}
                if llm_tracker_prefill is not None and llm_tracker_decode is not None:
                    prefill_metric = _extract_device_metric(llm_tracker_prefill)
                    decode_metric = _extract_device_metric(llm_tracker_decode)
                    prefill_time_series = _extract_device_time_series(llm_tracker_prefill)
                    decode_time_series = _extract_device_time_series(llm_tracker_decode)
                    llm_device_time_series = {"prefill": prefill_time_series, "decode": decode_time_series}
                    prefill_duration = float(getattr(llm_result, "prefill_phase_duration_s", 0.0) or 0.0)
                    decode_duration = float(getattr(llm_result, "decode_phase_duration_s", 0.0) or 0.0)
                    llm_result.avg_power_w = _weighted_two(
                        prefill_metric.get("avg_power_w"),
                        prefill_duration,
                        decode_metric.get("avg_power_w"),
                        decode_duration,
                    )
                    prefill_energy = _energy_from_device_time_series(prefill_time_series)
                    decode_energy = _energy_from_device_time_series(decode_time_series)
                    llm_energy = _sum_required_energies(prefill_energy, decode_energy)
                    llm_result.llm_prefill_energy_j = prefill_energy
                    llm_result.llm_decode_energy_j = decode_energy
                    llm_result.llm_total_energy_j = llm_energy
                    true_total_energy = _sum_required_energies(
                        current_vision_energy,
                        llm_energy,
                    )
                    llm_result.total_energy_j = true_total_energy
                    if prefill_energy is not None:
                        llm_prefill_energy_j.append(prefill_energy)
                    if decode_energy is not None:
                        llm_decode_energy_j.append(decode_energy)
                    if llm_energy is not None:
                        llm_total_energy_j.append(llm_energy)
                        prefill_tokens = _sweep_prefill_token_count(llm_result, batch_size)
                        decode_tokens = _sweep_decode_token_count(
                            llm_result,
                            decode_window=args.decode,
                            batch_size=batch_size,
                        )
                        total_tokens = prefill_tokens + decode_tokens
                        llm_result.prefill_tps_per_w = _safe_div(float(prefill_tokens), prefill_energy)
                        llm_result.decode_tps_per_w = _safe_div(float(decode_tokens), decode_energy)
                        llm_result.prefill_j_per_token = (
                            _safe_div(prefill_energy, float(prefill_tokens)) if prefill_tokens > 0 else None
                        )
                        llm_result.decode_j_per_token = (
                            _safe_div(decode_energy, float(decode_tokens)) if decode_tokens > 0 else None
                        )
                        llm_result.total_tps_per_w = _safe_div(float(total_tokens), llm_energy)
                        llm_result.total_j_per_token = (
                            _safe_div(llm_energy, float(total_tokens)) if total_tokens > 0 else None
                        )
                        if llm_result.prefill_tps_per_w is not None:
                            llm_prefill_tps_per_w.append(float(llm_result.prefill_tps_per_w))
                        if llm_result.decode_tps_per_w is not None:
                            llm_decode_tps_per_w.append(float(llm_result.decode_tps_per_w))
                    if true_total_energy is not None:
                        vlm_total_energy_j.append(true_total_energy)

                vision_runs.append({"vision_encode_latency": vision_latency_per_image, "vision_fps": vision_fps})
                llm_runs.append(_vlm_llm_run_payload(llm_result))
                device_time_series_runs.append({"vision": vision_device_time_series, "llm": llm_device_time_series})
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
            expected_runs = int(args.repeat)

            def _sum_complete(energies: Sequence[float]) -> float | None:
                """Return a repeat-total only when every measured repeat produced this energy."""
                return sum(float(energy) for energy in energies) if len(energies) == expected_runs else None

            vision_energy = _sum_complete(vision_energy_j)
            llm_prefill_energy = _sum_complete(llm_prefill_energy_j)
            llm_decode_energy = _sum_complete(llm_decode_energy_j)
            llm_total_energy = _sum_complete(llm_total_energy_j)
            total_energy = _sum_required_energies(vision_energy, llm_total_energy)
            payload = {
                "model": label,
                "benchmark_type": "measure",
                "task": "image-text-to-text",
                "batch_mode": batch_mode,
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
                    "vision_energy_j": vision_energy,
                    "llm_prefill_energy_j": llm_prefill_energy,
                    "llm_decode_energy_j": llm_decode_energy,
                    "llm_total_energy_j": llm_total_energy,
                    "total_energy_j": total_energy if len(vlm_total_energy_j) == expected_runs else None,
                    "llm_prefill_tps_per_w": _mean(llm_prefill_tps_per_w) if llm_prefill_tps_per_w else None,
                    "llm_decode_tps_per_w": _mean(llm_decode_tps_per_w) if llm_decode_tps_per_w else None,
                    "vision_img_per_j": _safe_div(len(vision_runs) * batch_size, vision_energy)
                    if vision_energy is not None
                    else None,
                }
                if avg_power_w or llm_total_energy_j or vision_energy_j
                else None,
                "device_time_series_runs": device_time_series_runs,
            }
            _write_json(json_path, payload)
            print(f"Saved: {json_path.name}")
        except Exception as e:
            print(f"Skipping {label} (measure failed): {e}")
        finally:
            _release_pipeline(pipeline, target_args.device)
    _rebuild_measure_outputs(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
