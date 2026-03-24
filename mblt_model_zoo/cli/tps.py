from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    add_device_tracking_args as _add_device_tracking_args,
    build_device_tracker as _build_device_tracker_common,
    build_phase_trackers as _build_phase_trackers_common,
    extract_device_metric as _extract_device_metric_common,
    parse_positive_int as _parse_positive_int_common,
    parse_positive_int_optional as _parse_positive_int_optional_common,
    print_device_status as _print_device_status_common,
    stop_tracker_safe as _stop_tracker_safe_common,
    weighted_two as _weighted_two_common,
)
from tqdm.auto import tqdm
from transformers import HfArgumentParser

def _parse_range(spec: str) -> Tuple[int, int, int]:
    text = spec.strip()
    sep = ":" if ":" in text else ("," if "," in text else None)
    if sep is None:
        raise argparse.ArgumentTypeError(
            f"invalid range '{spec}': expected 'start:end:step' or 'start,end,step'"
        )
    parts = [p.strip() for p in text.split(sep)]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"invalid range '{spec}': expected 3 integers (start, end, step)"
        )
    try:
        start, end, step = (int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid range '{spec}': start/end/step must be integers"
        ) from e
    if step <= 0:
        raise argparse.ArgumentTypeError(f"invalid range '{spec}': step must be > 0")
    if start <= 0 or end <= 0:
        raise argparse.ArgumentTypeError(
            f"invalid range '{spec}': start/end must be > 0"
        )
    if start > end:
        raise argparse.ArgumentTypeError(
            f"invalid range '{spec}': start must be <= end"
        )
    return start, end, step


def _parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("all values must be > 0")
    return values


def _parse_positive_int(spec: str) -> int:
    return _parse_positive_int_common(spec)


def _parse_positive_int_optional(spec: Union[str, None]) -> Union[int, None]:
    return _parse_positive_int_optional_common(spec)


def _parse_target_cores(spec: Union[str, None]) -> Union[list[str], None]:
    if spec is None:
        return None
    text = spec.strip()
    if not text:
        return None
    return [item.strip() for item in text.split(";") if item.strip()]


def _parse_target_clusters(spec: Union[str, None]) -> Union[list[int], None]:
    if spec is None:
        return None
    text = spec.strip()
    if not text:
        return None
    clusters: list[int] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        clusters.append(int(item))
    return clusters


def _require_transformers_deps() -> None:
    try:
        import transformers  # noqa: F401
    except Exception as e:
        print(
            "Missing optional dependencies for transformers TPS benchmarking.\n"
            "Install with: pip install 'mblt-model-zoo[transformers]'\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _build_pipeline(
    *,
    task: str,
    model: str,
    tokenizer: Union[str, None],
    device: str,
    trust_remote_code: bool,
    dtype: Union[str, None],
    device_map: Union[str, None],
    revision: Union[str, None],
    embedding_weight: Union[str, None],
    mxq_path: Union[str, None],
    core_mode: Union[str, None],
    target_cores: Union[list[str], None],
    target_clusters: Union[list[int], None],
) -> Any:
    _require_transformers_deps()
    from transformers import pipeline as hf_pipeline

    pipeline_kwargs: dict[str, Any] = {
        "task": task,
        "model": model,
        "trust_remote_code": trust_remote_code,
        "device": device,
    }
    if revision:
        pipeline_kwargs["revision"] = revision
    if tokenizer:
        pipeline_kwargs["tokenizer"] = tokenizer
    if device_map:
        pipeline_kwargs["device_map"] = device_map
    model_kwargs: dict[str, Any] = {}
    if embedding_weight:
        model_kwargs["embedding_weight"] = embedding_weight
    if mxq_path:
        model_kwargs["mxq_path"] = mxq_path
    if core_mode:
        model_kwargs["core_mode"] = core_mode
    if target_cores:
        model_kwargs["target_cores"] = target_cores
    if target_clusters:
        model_kwargs["target_clusters"] = target_clusters
    if model_kwargs:
        pipeline_kwargs["model_kwargs"] = model_kwargs

    def _raise_cuda_nvml_hint(exc: Exception) -> None:
        msg = str(exc)
        if "nvmlInit_v2" in msg or "Can't initialize NVML" in msg:
            raise SystemExit(
                "CUDA/NVML initialization failed while creating the pipeline.\n"
                "This happens before device tracking starts and is a host GPU driver/runtime issue.\n"
                "Check: `nvidia-smi`, `python -c \"import torch; print(torch.cuda.is_available())\"`.\n"
                "If running in container, verify NVIDIA runtime and libnvidia-ml visibility.\n"
                "Temporary workaround for this run: set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` and retry."
            ) from exc
        raise exc

    if dtype:
        try:
            pipeline_kwargs["dtype"] = dtype
            return hf_pipeline(**pipeline_kwargs)
        except TypeError:
            pipeline_kwargs.pop("dtype", None)
            pipeline_kwargs["torch_dtype"] = dtype
            try:
                return hf_pipeline(**pipeline_kwargs)
            except Exception as e:
                _raise_cuda_nvml_hint(e)
        except Exception as e:
            _raise_cuda_nvml_hint(e)

    try:
        return hf_pipeline(**pipeline_kwargs)
    except Exception as e:
        _raise_cuda_nvml_hint(e)


def _iter_rows_for_csv(result: Any) -> Iterable[dict[str, Any]]:
    for x, tps, t, avg_total, avg_npu in zip(
        result.prefill_sweep.x_values,
        result.prefill_sweep.tps_values,
        result.prefill_sweep.time_values,
        result.prefill_sweep.avg_total_token_latency_values,
        result.prefill_sweep.avg_npu_token_latency_values,
    ):
        yield {
            "phase": "prefill",
            "tokens": x,
            "tps": tps,
            "time_ms": t * 1000.0,
            "avg_total_token_latency_ms": avg_total * 1000.0 if avg_total is not None else None,
            "avg_npu_token_latency_ms": avg_npu * 1000.0 if avg_npu is not None else None,
        }
    for x, tps, t, avg_total, avg_npu in zip(
        result.decode_sweep.x_values,
        result.decode_sweep.tps_values,
        result.decode_sweep.time_values,
        result.decode_sweep.avg_total_token_latency_values,
        result.decode_sweep.avg_npu_token_latency_values,
    ):
        yield {
            "phase": "decode",
            "tokens": x,
            "tps": tps,
            "time_ms": t * 1000.0,
            "avg_total_token_latency_ms": avg_total * 1000.0 if avg_total is not None else None,
            "avg_npu_token_latency_ms": avg_npu * 1000.0 if avg_npu is not None else None,
        }


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _percentile(values: Sequence[float], q: float) -> float:
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
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "p50": _percentile(vals, 0.50),
        "p95": _percentile(vals, 0.95),
        "p99": _percentile(vals, 0.99),
    }


def _print_summary(name: str, values: Sequence[float], unit: str) -> None:
    s = _summary(values)
    print(
        f"| {name:<24} | {unit:<6} | {s['mean']:>10.3f} | {s['p50']:>10.3f} | "
        f"{s['p95']:>10.3f} | {s['p99']:>10.3f} | {s['min']:>10.3f} | {s['max']:>10.3f} |"
    )


def _print_summary_header() -> None:
    line = (
        "+--------------------------+--------+------------+------------+------------+"
        "------------+------------+------------+"
    )
    print(line)
    print(
        f"| {'metric':<24} | {'unit':<6} | {'mean':>10} | {'p50':>10} | "
        f"{'p95':>10} | {'p99':>10} | {'min':>10} | {'max':>10} |"
    )
    print(line)


def _print_summary_footer() -> None:
    print(
        "+--------------------------+--------+------------+------------+------------+"
        "------------+------------+------------+"
    )


def _build_device_tracker(args: argparse.Namespace, pipeline: Any):
    return _build_device_tracker_common(args, pipeline)


def _build_phase_trackers(args: argparse.Namespace, pipeline: Any) -> tuple[Any, Any]:
    return _build_phase_trackers_common(args, pipeline)


def _stop_tracker_safe(tracker: Any) -> None:
    _stop_tracker_safe_common(tracker)


def _extract_device_metric(tracker: Any) -> dict[str, Optional[float]]:
    return _extract_device_metric_common(tracker)


def _weighted_two(
    a: Optional[float],
    a_weight: float,
    b: Optional[float],
    b_weight: float,
) -> Optional[float]:
    return _weighted_two_common(a, a_weight, b, b_weight)


def _print_device_status(args: argparse.Namespace, tracker: Any) -> None:
    _print_device_status_common(args, tracker)


def _safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def _enrich_single_run_device(
    run: Any,
    prefill_metric: dict[str, Optional[float]],
    decode_metric: dict[str, Optional[float]],
) -> None:
    prefill_avg_power = prefill_metric.get("avg_power_w")
    decode_avg_power = decode_metric.get("avg_power_w")
    run.prefill_avg_power_w = prefill_avg_power
    run.prefill_p99_power_w = prefill_metric.get("p99_power_w")
    run.decode_avg_power_w = decode_avg_power
    run.decode_p99_power_w = decode_metric.get("p99_power_w")
    run.prefill_avg_utilization_pct = prefill_metric.get("avg_utilization_pct")
    run.prefill_p99_utilization_pct = prefill_metric.get("p99_utilization_pct")
    run.decode_avg_utilization_pct = decode_metric.get("avg_utilization_pct")
    run.decode_p99_utilization_pct = decode_metric.get("p99_utilization_pct")
    run.prefill_avg_memory_used_mb = prefill_metric.get("avg_memory_used_mb")
    run.prefill_p99_memory_used_mb = prefill_metric.get("p99_memory_used_mb")
    run.decode_avg_memory_used_mb = decode_metric.get("avg_memory_used_mb")
    run.decode_p99_memory_used_mb = decode_metric.get("p99_memory_used_mb")
    run.prefill_avg_memory_used_pct = prefill_metric.get("avg_memory_used_pct")
    run.prefill_p99_memory_used_pct = prefill_metric.get("p99_memory_used_pct")
    run.decode_avg_memory_used_pct = decode_metric.get("avg_memory_used_pct")
    run.decode_p99_memory_used_pct = decode_metric.get("p99_memory_used_pct")

    prefill_t = float(run.prefill_latency)
    decode_t = float(run.decode_duration)
    run.avg_power_w = _weighted_two(prefill_avg_power, prefill_t, decode_avg_power, decode_t)
    p_p99 = prefill_metric.get("p99_power_w")
    d_p99 = decode_metric.get("p99_power_w")
    run.p99_power_w = max([v for v in (p_p99, d_p99) if v is not None], default=None)
    run.avg_utilization_pct = _weighted_two(
        prefill_metric.get("avg_utilization_pct"),
        prefill_t,
        decode_metric.get("avg_utilization_pct"),
        decode_t,
    )
    p_u_p99 = prefill_metric.get("p99_utilization_pct")
    d_u_p99 = decode_metric.get("p99_utilization_pct")
    run.p99_utilization_pct = max([v for v in (p_u_p99, d_u_p99) if v is not None], default=None)
    run.avg_memory_used_mb = _weighted_two(
        prefill_metric.get("avg_memory_used_mb"),
        prefill_t,
        decode_metric.get("avg_memory_used_mb"),
        decode_t,
    )
    p_m_p99 = prefill_metric.get("p99_memory_used_mb")
    d_m_p99 = decode_metric.get("p99_memory_used_mb")
    run.p99_memory_used_mb = max([v for v in (p_m_p99, d_m_p99) if v is not None], default=None)
    run.avg_memory_used_pct = _weighted_two(
        prefill_metric.get("avg_memory_used_pct"),
        prefill_t,
        decode_metric.get("avg_memory_used_pct"),
        decode_t,
    )
    p_mp_p99 = prefill_metric.get("p99_memory_used_pct")
    d_mp_p99 = decode_metric.get("p99_memory_used_pct")
    run.p99_memory_used_pct = max([v for v in (p_mp_p99, d_mp_p99) if v is not None], default=None)
    run.total_memory_mb = max(
        [v for v in (prefill_metric.get("total_memory_mb"), decode_metric.get("total_memory_mb")) if v is not None],
        default=None,
    )

    avg_power = run.avg_power_w
    if avg_power is None:
        return

    prefill_energy = (
        prefill_avg_power * run.prefill_latency if prefill_avg_power is not None else None
    )
    decode_energy = (
        decode_avg_power * run.decode_duration if decode_avg_power is not None else None
    )
    total_energy = None
    if prefill_energy is not None and decode_energy is not None:
        total_energy = prefill_energy + decode_energy

    total_tokens = run.num_prefill + run.num_decode

    run.avg_power_w = float(avg_power)
    run.total_energy_j = total_energy
    run.prefill_tokens_per_j = (
        _safe_div(float(run.num_prefill), prefill_energy)
        if prefill_energy is not None
        else None
    )
    run.prefill_j_per_token = (
        _safe_div(prefill_energy, float(run.num_prefill))
        if prefill_energy is not None and run.num_prefill > 0
        else None
    )
    run.decode_tokens_per_j = (
        _safe_div(float(run.num_decode), decode_energy)
        if decode_energy is not None
        else None
    )
    run.decode_j_per_token = (
        _safe_div(decode_energy, float(run.num_decode))
        if decode_energy is not None and run.num_decode > 0
        else None
    )
    run.total_tokens_per_j = (
        _safe_div(float(total_tokens), total_energy) if total_energy is not None else None
    )
    run.total_j_per_token = (
        _safe_div(total_energy, float(total_tokens))
        if total_energy is not None and total_tokens > 0
        else None
    )


def _aggregate_sweep_results(results: Sequence[Any]) -> Any:
    if len(results) == 1:
        return results[0]

    from mblt_model_zoo.hf_transformers.utils.benchmark_utils import BenchmarkResult, SweepData

    def _mean_or_none(values: list[Union[float, None]]) -> Union[float, None]:
        compact = [float(v) for v in values if v is not None]
        if not compact:
            return None
        return sum(compact) / len(compact)

    def _aggregate_phase(phase: str) -> SweepData:
        first = results[0].prefill_sweep if phase == "prefill" else results[0].decode_sweep
        out = SweepData(x_values=list(first.x_values))
        for idx in range(len(first.x_values)):
            tps_vals = []
            time_vals = []
            total_vals = []
            npu_vals = []
            for result in results:
                src = result.prefill_sweep if phase == "prefill" else result.decode_sweep
                tps_vals.append(src.tps_values[idx])
                time_vals.append(src.time_values[idx])
                total_vals.append(src.avg_total_token_latency_values[idx])
                npu_vals.append(src.avg_npu_token_latency_values[idx])
            out.tps_values.append(sum(tps_vals) / len(tps_vals))
            out.time_values.append(sum(time_vals) / len(time_vals))
            out.avg_total_token_latency_values.append(_mean_or_none(total_vals))
            out.avg_npu_token_latency_values.append(_mean_or_none(npu_vals))
        return out

    return BenchmarkResult(
        prefill_sweep=_aggregate_phase("prefill"),
        decode_sweep=_aggregate_phase("decode"),
    )


def _cmd_measure(args: argparse.Namespace) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    pipeline = _build_pipeline(
        task=args.task,
        model=args.model,
        tokenizer=args.tokenizer,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device_map=args.device_map,
        revision=args.revision,
        embedding_weight=args.embedding_weight,
        mxq_path=args.mxq_path,
        core_mode=args.core_mode,
        target_cores=args.target_cores,
        target_clusters=args.target_clusters,
    )

    from mblt_model_zoo.hf_transformers.utils.benchmark_utils import TPSMeasurer

    measurer = TPSMeasurer(pipeline)
    tracker_prefill, tracker_decode = _build_phase_trackers(args, pipeline)
    _print_device_status(args, tracker_prefill)
    for i in tqdm(range(args.warmup), desc="warmup runs", leave=False):
        measurer.measure(
            num_prefill=args.prefill,
            num_decode=args.decode,
            chunk_size=args.chunk_size,
            trace_path=None,
            show_progress=True,
            progress_desc=f"warmup generate {i + 1}/{args.warmup}",
        )
    runs = []
    for i in tqdm(range(args.repeat), desc="measure runs", leave=False):
        prefill_metric: dict[str, Optional[float]] = {}
        decode_metric: dict[str, Optional[float]] = {}
        try:
            run = measurer.measure(
                num_prefill=args.prefill,
                num_decode=args.decode,
                chunk_size=args.chunk_size,
                trace_path=args.trace if i == 0 else None,
                show_progress=True,
                progress_desc=f"measure generate {i + 1}/{args.repeat}",
                on_prefill_start=(lambda: tracker_prefill.start()) if tracker_prefill is not None else None,
                on_prefill_end=(lambda: tracker_prefill.stop()) if tracker_prefill is not None else None,
                on_decode_start=(lambda: tracker_decode.start()) if tracker_decode is not None else None,
                on_decode_end=(lambda: tracker_decode.stop()) if tracker_decode is not None else None,
            )
        finally:
            _stop_tracker_safe(tracker_prefill)
            _stop_tracker_safe(tracker_decode)
        if tracker_prefill is not None and tracker_decode is not None:
            prefill_metric = _extract_device_metric(tracker_prefill)
            decode_metric = _extract_device_metric(tracker_decode)
            _enrich_single_run_device(
                run=run,
                prefill_metric=prefill_metric,
                decode_metric=decode_metric,
            )
        runs.append(run)

    prefill_tps = [r.prefill_tps for r in runs]
    decode_tps = [r.decode_tps for r in runs]
    ttft_ms = [r.prefill_latency * 1000.0 for r in runs]
    decode_ms = [r.decode_duration * 1000.0 for r in runs]
    total_ms = [r.total_time * 1000.0 for r in runs]
    avg_power_w = [r.avg_power_w for r in runs if r.avg_power_w is not None]
    p99_power_w = [r.p99_power_w for r in runs if r.p99_power_w is not None]
    avg_utilization_pct = [
        r.avg_utilization_pct for r in runs if r.avg_utilization_pct is not None
    ]
    p99_utilization_pct = [
        r.p99_utilization_pct for r in runs if r.p99_utilization_pct is not None
    ]
    avg_memory_used_mb = [
        r.avg_memory_used_mb for r in runs if r.avg_memory_used_mb is not None
    ]
    p99_memory_used_mb = [
        r.p99_memory_used_mb for r in runs if r.p99_memory_used_mb is not None
    ]
    total_memory_mb = [r.total_memory_mb for r in runs if r.total_memory_mb is not None]
    avg_memory_used_pct = [
        r.avg_memory_used_pct for r in runs if r.avg_memory_used_pct is not None
    ]
    p99_memory_used_pct = [
        r.p99_memory_used_pct for r in runs if r.p99_memory_used_pct is not None
    ]
    prefill_avg_power_w = [r.prefill_avg_power_w for r in runs if r.prefill_avg_power_w is not None]
    prefill_p99_power_w = [r.prefill_p99_power_w for r in runs if r.prefill_p99_power_w is not None]
    decode_avg_power_w = [r.decode_avg_power_w for r in runs if r.decode_avg_power_w is not None]
    decode_p99_power_w = [r.decode_p99_power_w for r in runs if r.decode_p99_power_w is not None]
    prefill_avg_utilization_pct = [
        r.prefill_avg_utilization_pct for r in runs if r.prefill_avg_utilization_pct is not None
    ]
    prefill_p99_utilization_pct = [
        r.prefill_p99_utilization_pct for r in runs if r.prefill_p99_utilization_pct is not None
    ]
    decode_avg_utilization_pct = [
        r.decode_avg_utilization_pct for r in runs if r.decode_avg_utilization_pct is not None
    ]
    decode_p99_utilization_pct = [
        r.decode_p99_utilization_pct for r in runs if r.decode_p99_utilization_pct is not None
    ]
    prefill_avg_memory_used_mb = [
        r.prefill_avg_memory_used_mb for r in runs if r.prefill_avg_memory_used_mb is not None
    ]
    prefill_p99_memory_used_mb = [
        r.prefill_p99_memory_used_mb for r in runs if r.prefill_p99_memory_used_mb is not None
    ]
    decode_avg_memory_used_mb = [
        r.decode_avg_memory_used_mb for r in runs if r.decode_avg_memory_used_mb is not None
    ]
    decode_p99_memory_used_mb = [
        r.decode_p99_memory_used_mb for r in runs if r.decode_p99_memory_used_mb is not None
    ]
    prefill_avg_memory_used_pct = [
        r.prefill_avg_memory_used_pct for r in runs if r.prefill_avg_memory_used_pct is not None
    ]
    prefill_p99_memory_used_pct = [
        r.prefill_p99_memory_used_pct for r in runs if r.prefill_p99_memory_used_pct is not None
    ]
    decode_avg_memory_used_pct = [
        r.decode_avg_memory_used_pct for r in runs if r.decode_avg_memory_used_pct is not None
    ]
    decode_p99_memory_used_pct = [
        r.decode_p99_memory_used_pct for r in runs if r.decode_p99_memory_used_pct is not None
    ]
    total_energy_j = [r.total_energy_j for r in runs if r.total_energy_j is not None]
    prefill_tok_per_j = [r.prefill_tokens_per_j for r in runs if r.prefill_tokens_per_j is not None]
    decode_tok_per_j = [r.decode_tokens_per_j for r in runs if r.decode_tokens_per_j is not None]
    prefill_j_per_tok = [r.prefill_j_per_token for r in runs if r.prefill_j_per_token is not None]
    decode_j_per_tok = [r.decode_j_per_token for r in runs if r.decode_j_per_token is not None]

    print(f"warmup: {args.warmup}")
    print(f"runs: {args.repeat}")
    print(f"prefill tokens: {runs[0].num_prefill} | decode tokens: {runs[0].num_decode}")
    _print_summary_header()
    _print_summary("prefill_tps", prefill_tps, "tok/s")
    _print_summary("decode_tps", decode_tps, "tok/s")
    _print_summary("ttft", ttft_ms, "ms")
    _print_summary("decode_duration", decode_ms, "ms")
    _print_summary("total", total_ms, "ms")
    if args.device_metrics:
        _print_summary("avg_power", avg_power_w, "W")
        _print_summary("p99_power", p99_power_w, "W")
        _print_summary("prefill_avg_power", prefill_avg_power_w, "W")
        _print_summary("prefill_p99_power", prefill_p99_power_w, "W")
        _print_summary("decode_avg_power", decode_avg_power_w, "W")
        _print_summary("decode_p99_power", decode_p99_power_w, "W")
        _print_summary("avg_utilization", avg_utilization_pct, "%")
        _print_summary("p99_utilization", p99_utilization_pct, "%")
        _print_summary("prefill_avg_util", prefill_avg_utilization_pct, "%")
        _print_summary("prefill_p99_util", prefill_p99_utilization_pct, "%")
        _print_summary("decode_avg_util", decode_avg_utilization_pct, "%")
        _print_summary("decode_p99_util", decode_p99_utilization_pct, "%")
        _print_summary("avg_memory_used", avg_memory_used_mb, "MB")
        _print_summary("p99_memory_used", p99_memory_used_mb, "MB")
        _print_summary("prefill_avg_mem_used", prefill_avg_memory_used_mb, "MB")
        _print_summary("prefill_p99_mem_used", prefill_p99_memory_used_mb, "MB")
        _print_summary("decode_avg_mem_used", decode_avg_memory_used_mb, "MB")
        _print_summary("decode_p99_mem_used", decode_p99_memory_used_mb, "MB")
        _print_summary("total_memory", total_memory_mb, "MB")
        _print_summary("avg_memory_used_pct", avg_memory_used_pct, "%")
        _print_summary("p99_memory_used_pct", p99_memory_used_pct, "%")
        _print_summary("prefill_avg_mem_used_pct", prefill_avg_memory_used_pct, "%")
        _print_summary("prefill_p99_mem_used_pct", prefill_p99_memory_used_pct, "%")
        _print_summary("decode_avg_mem_used_pct", decode_avg_memory_used_pct, "%")
        _print_summary("decode_p99_mem_used_pct", decode_p99_memory_used_pct, "%")
        _print_summary("total_energy", total_energy_j, "J")
        _print_summary("prefill_tok_per_j", prefill_tok_per_j, "tok/J")
        _print_summary("decode_tok_per_j", decode_tok_per_j, "tok/J")
        _print_summary("prefill_j_per_tok", prefill_j_per_tok, "J/tok")
        _print_summary("decode_j_per_tok", decode_j_per_tok, "J/tok")
    _print_summary_footer()

    if args.json:
        payload = {
            "repeat": args.repeat,
            "runs": [asdict(r) for r in runs],
            "summary": {
                "prefill_tps": _summary(prefill_tps),
                "decode_tps": _summary(decode_tps),
                "ttft_ms": _summary(ttft_ms),
                "decode_duration_ms": _summary(decode_ms),
                "total_ms": _summary(total_ms),
                "avg_power_w": _summary(avg_power_w),
                "p99_power_w": _summary(p99_power_w),
                "prefill_avg_power_w": _summary(prefill_avg_power_w),
                "prefill_p99_power_w": _summary(prefill_p99_power_w),
                "decode_avg_power_w": _summary(decode_avg_power_w),
                "decode_p99_power_w": _summary(decode_p99_power_w),
                "avg_utilization_pct": _summary(avg_utilization_pct),
                "p99_utilization_pct": _summary(p99_utilization_pct),
                "prefill_avg_utilization_pct": _summary(prefill_avg_utilization_pct),
                "prefill_p99_utilization_pct": _summary(prefill_p99_utilization_pct),
                "decode_avg_utilization_pct": _summary(decode_avg_utilization_pct),
                "decode_p99_utilization_pct": _summary(decode_p99_utilization_pct),
                "avg_memory_used_mb": _summary(avg_memory_used_mb),
                "p99_memory_used_mb": _summary(p99_memory_used_mb),
                "prefill_avg_memory_used_mb": _summary(prefill_avg_memory_used_mb),
                "prefill_p99_memory_used_mb": _summary(prefill_p99_memory_used_mb),
                "decode_avg_memory_used_mb": _summary(decode_avg_memory_used_mb),
                "decode_p99_memory_used_mb": _summary(decode_p99_memory_used_mb),
                "total_memory_mb": _summary(total_memory_mb),
                "avg_memory_used_pct": _summary(avg_memory_used_pct),
                "p99_memory_used_pct": _summary(p99_memory_used_pct),
                "prefill_avg_memory_used_pct": _summary(prefill_avg_memory_used_pct),
                "prefill_p99_memory_used_pct": _summary(prefill_p99_memory_used_pct),
                "decode_avg_memory_used_pct": _summary(decode_avg_memory_used_pct),
                "decode_p99_memory_used_pct": _summary(decode_p99_memory_used_pct),
                "total_energy_j": _summary(total_energy_j),
                "prefill_tok_per_j": _summary(prefill_tok_per_j),
                "decode_tok_per_j": _summary(decode_tok_per_j),
                "prefill_j_per_tok": _summary(prefill_j_per_tok),
                "decode_j_per_tok": _summary(decode_j_per_tok),
            },
        }
        _write_json(args.json, payload)
        print(f"wrote: {args.json}")

    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    pipeline = _build_pipeline(
        task=args.task,
        model=args.model,
        tokenizer=args.tokenizer,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device_map=args.device_map,
        revision=args.revision,
        embedding_weight=args.embedding_weight,
        mxq_path=args.mxq_path,
        core_mode=args.core_mode,
        target_cores=args.target_cores,
        target_clusters=args.target_clusters,
    )

    from mblt_model_zoo.hf_transformers.utils.benchmark_utils import TPSMeasurer

    measurer = TPSMeasurer(pipeline)
    tracker_prefill, tracker_decode = _build_phase_trackers(args, pipeline)
    _print_device_status(args, tracker_prefill)
    for i in tqdm(range(args.warmup), desc="warmup runs", leave=False):
        measurer.measure(
            num_prefill=args.fixed_prefill,
            num_decode=args.fixed_decode,
            chunk_size=args.chunk_size,
            trace_path=None,
            show_progress=True,
            progress_desc=f"warmup generate {i + 1}/{args.warmup}",
        )
    runs = []
    run_avg_power: list[float] = []
    run_p99_power: list[float] = []
    run_avg_utilization: list[float] = []
    run_p99_utilization: list[float] = []
    run_avg_memory_used_mb: list[float] = []
    run_p99_memory_used_mb: list[float] = []
    run_total_memory_mb: list[float] = []
    run_avg_memory_used_pct: list[float] = []
    run_p99_memory_used_pct: list[float] = []
    run_total_energy: list[float] = []
    run_prefill_avg_power: list[float] = []
    run_prefill_p99_power: list[float] = []
    run_decode_avg_power: list[float] = []
    run_decode_p99_power: list[float] = []
    run_prefill_avg_util: list[float] = []
    run_prefill_p99_util: list[float] = []
    run_decode_avg_util: list[float] = []
    run_decode_p99_util: list[float] = []
    run_prefill_avg_mem_used_mb: list[float] = []
    run_prefill_p99_mem_used_mb: list[float] = []
    run_decode_avg_mem_used_mb: list[float] = []
    run_decode_p99_mem_used_mb: list[float] = []
    run_prefill_avg_mem_used_pct: list[float] = []
    run_prefill_p99_mem_used_pct: list[float] = []
    run_decode_avg_mem_used_pct: list[float] = []
    run_decode_p99_mem_used_pct: list[float] = []
    run_phase_device: list[dict[str, dict[str, Optional[float]]]] = []
    for i in tqdm(range(args.repeat), desc="sweep runs", leave=False):
        prefill_metric: dict[str, Optional[float]] = {}
        decode_metric: dict[str, Optional[float]] = {}
        try:
            runs.append(
                measurer.measure_full(
                    prefill_range=args.prefill_range,
                    decode_range=args.decode_range,
                    fixed_decode_len=args.fixed_decode,
                    fixed_prefill_len=args.fixed_prefill,
                    chunk_size=args.chunk_size,
                    trace_path=args.trace if i == 0 else None,
                    show_progress=True,
                    progress_prefix=f"run {i + 1}/{args.repeat}",
                    on_prefill_start=(lambda: tracker_prefill.start()) if tracker_prefill is not None else None,
                    on_prefill_end=(lambda: tracker_prefill.stop()) if tracker_prefill is not None else None,
                    on_decode_start=(lambda: tracker_decode.start()) if tracker_decode is not None else None,
                    on_decode_end=(lambda: tracker_decode.stop()) if tracker_decode is not None else None,
                )
            )
        finally:
            _stop_tracker_safe(tracker_prefill)
            _stop_tracker_safe(tracker_decode)
        if tracker_prefill is not None and tracker_decode is not None:
            prefill_metric = _extract_device_metric(tracker_prefill)
            decode_metric = _extract_device_metric(tracker_decode)
            run_phase_device.append({"prefill": prefill_metric, "decode": decode_metric})
            prefill_dur = float(getattr(runs[-1], "prefill_phase_duration_s", 0.0) or 0.0)
            decode_dur = float(getattr(runs[-1], "decode_phase_duration_s", 0.0) or 0.0)
            avg_power = _weighted_two(
                prefill_metric.get("avg_power_w"),
                prefill_dur,
                decode_metric.get("avg_power_w"),
                decode_dur,
            )
            if avg_power is not None:
                run_avg_power.append(avg_power)
                total_energy = avg_power * (prefill_dur + decode_dur)
                run_total_energy.append(total_energy)
            p99_power = max(
                [v for v in (prefill_metric.get("p99_power_w"), decode_metric.get("p99_power_w")) if v is not None],
                default=None,
            )
            if p99_power is not None:
                run_p99_power.append(float(p99_power))
            avg_util = _weighted_two(
                prefill_metric.get("avg_utilization_pct"),
                prefill_dur,
                decode_metric.get("avg_utilization_pct"),
                decode_dur,
            )
            if avg_util is not None:
                run_avg_utilization.append(float(avg_util))
            p99_util = max(
                [
                    v
                    for v in (
                        prefill_metric.get("p99_utilization_pct"),
                        decode_metric.get("p99_utilization_pct"),
                    )
                    if v is not None
                ],
                default=None,
            )
            if p99_util is not None:
                run_p99_utilization.append(float(p99_util))
            avg_mem_used_mb = _weighted_two(
                prefill_metric.get("avg_memory_used_mb"),
                prefill_dur,
                decode_metric.get("avg_memory_used_mb"),
                decode_dur,
            )
            if avg_mem_used_mb is not None:
                run_avg_memory_used_mb.append(float(avg_mem_used_mb))
            p99_mem_used_mb = max(
                [
                    v
                    for v in (
                        prefill_metric.get("p99_memory_used_mb"),
                        decode_metric.get("p99_memory_used_mb"),
                    )
                    if v is not None
                ],
                default=None,
            )
            if p99_mem_used_mb is not None:
                run_p99_memory_used_mb.append(float(p99_mem_used_mb))
            total_memory_mb = max(
                [v for v in (prefill_metric.get("total_memory_mb"), decode_metric.get("total_memory_mb")) if v is not None],
                default=None,
            )
            if total_memory_mb is not None:
                run_total_memory_mb.append(float(total_memory_mb))
            avg_mem_used_pct = _weighted_two(
                prefill_metric.get("avg_memory_used_pct"),
                prefill_dur,
                decode_metric.get("avg_memory_used_pct"),
                decode_dur,
            )
            if avg_mem_used_pct is not None:
                run_avg_memory_used_pct.append(float(avg_mem_used_pct))
            p99_mem_used_pct = max(
                [
                    v
                    for v in (
                        prefill_metric.get("p99_memory_used_pct"),
                        decode_metric.get("p99_memory_used_pct"),
                    )
                    if v is not None
                ],
                default=None,
            )
            if p99_mem_used_pct is not None:
                run_p99_memory_used_pct.append(float(p99_mem_used_pct))
            for dst, key in (
                (run_prefill_avg_power, "avg_power_w"),
                (run_prefill_p99_power, "p99_power_w"),
                (run_prefill_avg_util, "avg_utilization_pct"),
                (run_prefill_p99_util, "p99_utilization_pct"),
                (run_prefill_avg_mem_used_mb, "avg_memory_used_mb"),
                (run_prefill_p99_mem_used_mb, "p99_memory_used_mb"),
                (run_prefill_avg_mem_used_pct, "avg_memory_used_pct"),
                (run_prefill_p99_mem_used_pct, "p99_memory_used_pct"),
            ):
                v = prefill_metric.get(key)
                if isinstance(v, (int, float)):
                    dst.append(float(v))
            for dst, key in (
                (run_decode_avg_power, "avg_power_w"),
                (run_decode_p99_power, "p99_power_w"),
                (run_decode_avg_util, "avg_utilization_pct"),
                (run_decode_p99_util, "p99_utilization_pct"),
                (run_decode_avg_mem_used_mb, "avg_memory_used_mb"),
                (run_decode_p99_mem_used_mb, "p99_memory_used_mb"),
                (run_decode_avg_mem_used_pct, "avg_memory_used_pct"),
                (run_decode_p99_mem_used_pct, "p99_memory_used_pct"),
            ):
                v = decode_metric.get(key)
                if isinstance(v, (int, float)):
                    dst.append(float(v))

    result = _aggregate_sweep_results(runs)

    prefill_last = [r.prefill_sweep.tps_values[-1] for r in runs if r.prefill_sweep.tps_values]
    decode_last = [r.decode_sweep.tps_values[-1] for r in runs if r.decode_sweep.tps_values]
    prefill_last_tpj: list[float] = []
    decode_last_tpj: list[float] = []
    prefill_last_jpt: list[float] = []
    decode_last_jpt: list[float] = []
    for i, p_tps in enumerate(prefill_last):
        phase_power = None
        if i < len(run_phase_device):
            v = run_phase_device[i].get("prefill", {}).get("avg_power_w")
            if isinstance(v, (int, float)):
                phase_power = float(v)
        if phase_power is not None and phase_power > 0:
            tpj = p_tps / phase_power
            prefill_last_tpj.append(tpj)
            prefill_last_jpt.append(1.0 / tpj if tpj > 0 else 0.0)
    for i, d_tps in enumerate(decode_last):
        phase_power = None
        if i < len(run_phase_device):
            v = run_phase_device[i].get("decode", {}).get("avg_power_w")
            if isinstance(v, (int, float)):
                phase_power = float(v)
        if phase_power is not None and phase_power > 0:
            tpj = d_tps / phase_power
            decode_last_tpj.append(tpj)
            decode_last_jpt.append(1.0 / tpj if tpj > 0 else 0.0)
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.repeat}")
    _print_summary_header()
    if prefill_last:
        _print_summary("prefill_tps(last_point)", prefill_last, "tok/s")
    if decode_last:
        _print_summary("decode_tps(last_point)", decode_last, "tok/s")
    if args.device_metrics:
        _print_summary("avg_power", run_avg_power, "W")
        _print_summary("p99_power", run_p99_power, "W")
        _print_summary("prefill_avg_power", run_prefill_avg_power, "W")
        _print_summary("prefill_p99_power", run_prefill_p99_power, "W")
        _print_summary("decode_avg_power", run_decode_avg_power, "W")
        _print_summary("decode_p99_power", run_decode_p99_power, "W")
        _print_summary("avg_utilization", run_avg_utilization, "%")
        _print_summary("p99_utilization", run_p99_utilization, "%")
        _print_summary("prefill_avg_util", run_prefill_avg_util, "%")
        _print_summary("prefill_p99_util", run_prefill_p99_util, "%")
        _print_summary("decode_avg_util", run_decode_avg_util, "%")
        _print_summary("decode_p99_util", run_decode_p99_util, "%")
        _print_summary("avg_memory_used", run_avg_memory_used_mb, "MB")
        _print_summary("p99_memory_used", run_p99_memory_used_mb, "MB")
        _print_summary("prefill_avg_mem_used", run_prefill_avg_mem_used_mb, "MB")
        _print_summary("prefill_p99_mem_used", run_prefill_p99_mem_used_mb, "MB")
        _print_summary("decode_avg_mem_used", run_decode_avg_mem_used_mb, "MB")
        _print_summary("decode_p99_mem_used", run_decode_p99_mem_used_mb, "MB")
        _print_summary("total_memory", run_total_memory_mb, "MB")
        _print_summary("avg_memory_used_pct", run_avg_memory_used_pct, "%")
        _print_summary("p99_memory_used_pct", run_p99_memory_used_pct, "%")
        _print_summary("prefill_avg_mem_used_pct", run_prefill_avg_mem_used_pct, "%")
        _print_summary("prefill_p99_mem_used_pct", run_prefill_p99_mem_used_pct, "%")
        _print_summary("decode_avg_mem_used_pct", run_decode_avg_mem_used_pct, "%")
        _print_summary("decode_p99_mem_used_pct", run_decode_p99_mem_used_pct, "%")
        _print_summary("total_energy", run_total_energy, "J")
        _print_summary("prefill_tok_per_j(last)", prefill_last_tpj, "tok/J")
        _print_summary("decode_tok_per_j(last)", decode_last_tpj, "tok/J")
        _print_summary("prefill_j_per_tok(last)", prefill_last_jpt, "J/tok")
        _print_summary("decode_j_per_tok(last)", decode_last_jpt, "J/tok")
    _print_summary_footer()

    if args.json:
        payload = {
            "repeat": args.repeat,
            "aggregate": asdict(result),
            "runs": [asdict(r) for r in runs],
            "summary": {
                "prefill_tps_last": _summary(prefill_last),
                "decode_tps_last": _summary(decode_last),
                "avg_power_w": _summary(run_avg_power),
                "p99_power_w": _summary(run_p99_power),
                "avg_utilization_pct": _summary(run_avg_utilization),
                "p99_utilization_pct": _summary(run_p99_utilization),
                "prefill_avg_power_w": _summary(run_prefill_avg_power),
                "prefill_p99_power_w": _summary(run_prefill_p99_power),
                "decode_avg_power_w": _summary(run_decode_avg_power),
                "decode_p99_power_w": _summary(run_decode_p99_power),
                "prefill_avg_utilization_pct": _summary(run_prefill_avg_util),
                "prefill_p99_utilization_pct": _summary(run_prefill_p99_util),
                "decode_avg_utilization_pct": _summary(run_decode_avg_util),
                "decode_p99_utilization_pct": _summary(run_decode_p99_util),
                "avg_memory_used_mb": _summary(run_avg_memory_used_mb),
                "p99_memory_used_mb": _summary(run_p99_memory_used_mb),
                "prefill_avg_memory_used_mb": _summary(run_prefill_avg_mem_used_mb),
                "prefill_p99_memory_used_mb": _summary(run_prefill_p99_mem_used_mb),
                "decode_avg_memory_used_mb": _summary(run_decode_avg_mem_used_mb),
                "decode_p99_memory_used_mb": _summary(run_decode_p99_mem_used_mb),
                "total_memory_mb": _summary(run_total_memory_mb),
                "avg_memory_used_pct": _summary(run_avg_memory_used_pct),
                "p99_memory_used_pct": _summary(run_p99_memory_used_pct),
                "prefill_avg_memory_used_pct": _summary(run_prefill_avg_mem_used_pct),
                "prefill_p99_memory_used_pct": _summary(run_prefill_p99_mem_used_pct),
                "decode_avg_memory_used_pct": _summary(run_decode_avg_mem_used_pct),
                "decode_p99_memory_used_pct": _summary(run_decode_p99_mem_used_pct),
                "total_energy_j": _summary(run_total_energy),
                "prefill_tok_per_j_last": _summary(prefill_last_tpj),
                "decode_tok_per_j_last": _summary(decode_last_tpj),
                "prefill_j_per_tok_last": _summary(prefill_last_jpt),
                "decode_j_per_tok_last": _summary(decode_last_jpt),
            },
            "device_runs": run_phase_device,
        }
        _write_json(args.json, payload)
        print(f"wrote: {args.json}")

    if args.csv:
        rows = list(_iter_rows_for_csv(result))
        _write_csv(args.csv, rows)
        print(f"wrote: {args.csv}")

    if args.plot:
        measurer.plot_and_save(result, save_path=args.plot)

    return 0


def _cmd_vlm_sweep(args: argparse.Namespace) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    pipeline = _build_pipeline(
        task=args.task,
        model=args.model,
        tokenizer=args.tokenizer,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device_map=args.device_map,
        revision=args.revision,
        embedding_weight=args.embedding_weight,
        mxq_path=args.mxq_path,
        core_mode=args.core_mode,
        target_cores=args.target_cores,
        target_clusters=args.target_clusters,
    )

    from mblt_model_zoo.hf_transformers.utils.benchmark_utils import VLMTPSMeasurer

    measurer = VLMTPSMeasurer(pipeline)
    tracker = _build_device_tracker(args, pipeline)
    _print_device_status(args, tracker)

    resolution_payloads = []
    csv_rows: list[dict[str, Any]] = []
    for resolution in args.image_resolutions:
        for _ in range(args.warmup):
            measurer.measure_vision(
                image_resolution=resolution,
                repeat=1,
                prompt=args.prompt,
                show_progress=False,
            )
        vision_runs = []
        vision_power_avg = []
        vision_power_p99 = []
        vision_util_avg = []
        vision_util_p99 = []
        vision_mem_used_avg_mb = []
        vision_mem_used_p99_mb = []
        vision_mem_total_mb = []
        vision_mem_used_pct_avg = []
        vision_mem_used_pct_p99 = []
        vision_energy_j = []
        vision_img_per_j = []
        vision_j_per_img = []
        for _ in tqdm(range(args.repeat), desc=f"vision@{resolution}", leave=False):
            if tracker is not None:
                tracker.start()
            try:
                single = measurer.measure_vision(
                    image_resolution=resolution,
                    repeat=1,
                    prompt=args.prompt,
                    show_progress=False,
                )[0]
            finally:
                if tracker is not None:
                    tracker.stop()
            vision_runs.append(single)
            if tracker is not None:
                metric = tracker.get_metric()
                avg_power = metric.get("avg_power_w")
                p99_power = metric.get("p99_power_w")
                avg_utilization = metric.get("avg_utilization_pct")
                p99_utilization = metric.get("p99_utilization_pct")
                avg_memory_used_mb = metric.get("avg_memory_used_mb")
                p99_memory_used_mb = metric.get("p99_memory_used_mb")
                total_memory_mb = metric.get("total_memory_mb")
                avg_memory_used_pct = metric.get("avg_memory_used_pct")
                p99_memory_used_pct = metric.get("p99_memory_used_pct")
                if avg_power is not None:
                    avg_power_f = float(avg_power)
                    energy = avg_power_f * float(single[0])
                    vision_power_avg.append(avg_power_f)
                    vision_energy_j.append(energy)
                    vision_img_per_j.append(1.0 / energy if energy > 0 else 0.0)
                    vision_j_per_img.append(energy)
                if p99_power is not None:
                    vision_power_p99.append(float(p99_power))
                if avg_utilization is not None:
                    vision_util_avg.append(float(avg_utilization))
                if p99_utilization is not None:
                    vision_util_p99.append(float(p99_utilization))
                if avg_memory_used_mb is not None:
                    vision_mem_used_avg_mb.append(float(avg_memory_used_mb))
                if p99_memory_used_mb is not None:
                    vision_mem_used_p99_mb.append(float(p99_memory_used_mb))
                if total_memory_mb is not None:
                    vision_mem_total_mb.append(float(total_memory_mb))
                if avg_memory_used_pct is not None:
                    vision_mem_used_pct_avg.append(float(avg_memory_used_pct))
                if p99_memory_used_pct is not None:
                    vision_mem_used_pct_p99.append(float(p99_memory_used_pct))

        vision_ms = [lat * 1000.0 for lat, _ in vision_runs]
        vision_fps = [fps for _, fps in vision_runs]

        print(f"\nresolution={resolution} warmup={args.warmup} runs={args.repeat}")
        _print_summary_header()
        _print_summary("vision_encode", vision_ms, "ms")
        _print_summary("vision_fps", vision_fps, "fps")
        if args.device_metrics:
            _print_summary("vision_avg_power", vision_power_avg, "W")
            _print_summary("vision_p99_power", vision_power_p99, "W")
            _print_summary("vision_avg_util", vision_util_avg, "%")
            _print_summary("vision_p99_util", vision_util_p99, "%")
            _print_summary("vision_avg_mem_used", vision_mem_used_avg_mb, "MB")
            _print_summary("vision_p99_mem_used", vision_mem_used_p99_mb, "MB")
            _print_summary("vision_total_mem", vision_mem_total_mb, "MB")
            _print_summary("vision_avg_mem_used_pct", vision_mem_used_pct_avg, "%")
            _print_summary("vision_p99_mem_used_pct", vision_mem_used_pct_p99, "%")
            _print_summary("vision_energy", vision_energy_j, "J")
            _print_summary("vision_img_per_j", vision_img_per_j, "img/J")
            _print_summary("vision_j_per_img", vision_j_per_img, "J/img")
            if not vision_power_avg:
                print("[device] warning: no vision device samples were collected for this resolution")
        _print_summary_footer()

        for idx, (latency, fps) in enumerate(vision_runs, start=1):
            csv_rows.append(
                {
                    "type": "vision",
                    "image_resolution": resolution,
                    "repeat_index": idx,
                    "vision_encode_ms": latency * 1000.0,
                    "vision_fps": fps,
                    "llm_prefill_tokens": None,
                    "llm_decode_tokens": None,
                    "llm_prefill_tps": None,
                    "llm_decode_tps": None,
                    "llm_ttft_ms": None,
                    "llm_decode_ms": None,
                    "llm_total_ms": None,
                    "avg_power_w": vision_power_avg[idx - 1] if idx - 1 < len(vision_power_avg) else None,
                    "p99_power_w": vision_power_p99[idx - 1] if idx - 1 < len(vision_power_p99) else None,
                    "avg_utilization_pct": vision_util_avg[idx - 1] if idx - 1 < len(vision_util_avg) else None,
                    "p99_utilization_pct": vision_util_p99[idx - 1] if idx - 1 < len(vision_util_p99) else None,
                    "avg_memory_used_mb": vision_mem_used_avg_mb[idx - 1] if idx - 1 < len(vision_mem_used_avg_mb) else None,
                    "p99_memory_used_mb": vision_mem_used_p99_mb[idx - 1] if idx - 1 < len(vision_mem_used_p99_mb) else None,
                    "total_memory_mb": vision_mem_total_mb[idx - 1] if idx - 1 < len(vision_mem_total_mb) else None,
                    "avg_memory_used_pct": vision_mem_used_pct_avg[idx - 1] if idx - 1 < len(vision_mem_used_pct_avg) else None,
                    "p99_memory_used_pct": vision_mem_used_pct_p99[idx - 1] if idx - 1 < len(vision_mem_used_pct_p99) else None,
                    "total_energy_j": vision_energy_j[idx - 1] if idx - 1 < len(vision_energy_j) else None,
                    "prefill_tok_per_j": None,
                    "decode_tok_per_j": None,
                    "prefill_j_per_tok": None,
                    "decode_j_per_tok": None,
                    "vision_img_per_j": vision_img_per_j[idx - 1] if idx - 1 < len(vision_img_per_j) else None,
                    "vision_j_per_img": vision_j_per_img[idx - 1] if idx - 1 < len(vision_j_per_img) else None,
                }
            )

        resolution_payloads.append(
            {
                "image_resolution": resolution,
                "repeat": args.repeat,
                "runs": [
                    {
                        "vision_encode_latency": latency,
                        "vision_fps": fps,
                    }
                    for latency, fps in vision_runs
                ],
                "summary": {
                    "vision_encode_ms": _summary(vision_ms),
                    "vision_fps": _summary(vision_fps),
                    "avg_power_w": _summary(vision_power_avg),
                    "p99_power_w": _summary(vision_power_p99),
                    "avg_utilization_pct": _summary(vision_util_avg),
                    "p99_utilization_pct": _summary(vision_util_p99),
                    "avg_memory_used_mb": _summary(vision_mem_used_avg_mb),
                    "p99_memory_used_mb": _summary(vision_mem_used_p99_mb),
                    "total_memory_mb": _summary(vision_mem_total_mb),
                    "avg_memory_used_pct": _summary(vision_mem_used_pct_avg),
                    "p99_memory_used_pct": _summary(vision_mem_used_pct_p99),
                    "energy_j": _summary(vision_energy_j),
                    "vision_img_per_j": _summary(vision_img_per_j),
                    "vision_j_per_img": _summary(vision_j_per_img),
                },
            }
        )

    llm_resolution = (
        args.llm_resolution
        if args.llm_resolution is not None
        else args.image_resolutions[0]
    )
    for _ in range(args.warmup):
        measurer.measure_llm(
            image_resolution=llm_resolution,
            num_decode=args.decode,
            repeat=1,
            prompt=args.prompt,
            show_progress=False,
        )
    llm_runs = []
    for _ in tqdm(range(args.repeat), desc=f"llm@{llm_resolution}", leave=False):
        if tracker is not None:
            tracker.start()
        try:
            run = measurer.measure_llm(
                image_resolution=llm_resolution,
                num_decode=args.decode,
                repeat=1,
                prompt=args.prompt,
                show_progress=False,
            )[0]
        finally:
            if tracker is not None:
                tracker.stop()
        if tracker is not None:
            metric = _extract_device_metric(tracker)
            _enrich_single_run_device(
                run=run,
                prefill_metric=metric,
                decode_metric=metric,
            )
        llm_runs.append(run)

    llm_prefill_tps = [r.prefill_tps for r in llm_runs]
    llm_decode_tps = [r.decode_tps for r in llm_runs]
    llm_ttft_ms = [r.prefill_latency * 1000.0 for r in llm_runs]
    llm_decode_ms = [r.decode_duration * 1000.0 for r in llm_runs]
    llm_total_ms = [r.total_time * 1000.0 for r in llm_runs]
    llm_avg_power_w = [r.avg_power_w for r in llm_runs if r.avg_power_w is not None]
    llm_p99_power_w = [r.p99_power_w for r in llm_runs if r.p99_power_w is not None]
    llm_avg_utilization_pct = [
        r.avg_utilization_pct for r in llm_runs if r.avg_utilization_pct is not None
    ]
    llm_p99_utilization_pct = [
        r.p99_utilization_pct for r in llm_runs if r.p99_utilization_pct is not None
    ]
    llm_avg_memory_used_mb = [
        r.avg_memory_used_mb for r in llm_runs if r.avg_memory_used_mb is not None
    ]
    llm_p99_memory_used_mb = [
        r.p99_memory_used_mb for r in llm_runs if r.p99_memory_used_mb is not None
    ]
    llm_total_memory_mb = [r.total_memory_mb for r in llm_runs if r.total_memory_mb is not None]
    llm_avg_memory_used_pct = [
        r.avg_memory_used_pct for r in llm_runs if r.avg_memory_used_pct is not None
    ]
    llm_p99_memory_used_pct = [
        r.p99_memory_used_pct for r in llm_runs if r.p99_memory_used_pct is not None
    ]
    llm_total_energy_j = [r.total_energy_j for r in llm_runs if r.total_energy_j is not None]
    llm_prefill_tok_per_j = [r.prefill_tokens_per_j for r in llm_runs if r.prefill_tokens_per_j is not None]
    llm_decode_tok_per_j = [r.decode_tokens_per_j for r in llm_runs if r.decode_tokens_per_j is not None]
    llm_prefill_j_per_tok = [r.prefill_j_per_token for r in llm_runs if r.prefill_j_per_token is not None]
    llm_decode_j_per_tok = [r.decode_j_per_token for r in llm_runs if r.decode_j_per_token is not None]

    print(
        f"\nllm_reference_resolution={llm_resolution} warmup={args.warmup} runs={args.repeat}"
    )
    _print_summary_header()
    _print_summary("llm_prefill_tps", llm_prefill_tps, "tok/s")
    _print_summary("llm_decode_tps", llm_decode_tps, "tok/s")
    _print_summary("llm_ttft", llm_ttft_ms, "ms")
    _print_summary("llm_decode_duration", llm_decode_ms, "ms")
    _print_summary("llm_total", llm_total_ms, "ms")
    if args.device_metrics:
        _print_summary("llm_avg_power", llm_avg_power_w, "W")
        _print_summary("llm_p99_power", llm_p99_power_w, "W")
        _print_summary("llm_avg_utilization", llm_avg_utilization_pct, "%")
        _print_summary("llm_p99_utilization", llm_p99_utilization_pct, "%")
        _print_summary("llm_avg_mem_used", llm_avg_memory_used_mb, "MB")
        _print_summary("llm_p99_mem_used", llm_p99_memory_used_mb, "MB")
        _print_summary("llm_total_mem", llm_total_memory_mb, "MB")
        _print_summary("llm_avg_mem_used_pct", llm_avg_memory_used_pct, "%")
        _print_summary("llm_p99_mem_used_pct", llm_p99_memory_used_pct, "%")
        _print_summary("llm_total_energy", llm_total_energy_j, "J")
        _print_summary("llm_prefill_tok_per_j", llm_prefill_tok_per_j, "tok/J")
        _print_summary("llm_decode_tok_per_j", llm_decode_tok_per_j, "tok/J")
        _print_summary("llm_prefill_j_per_tok", llm_prefill_j_per_tok, "J/tok")
        _print_summary("llm_decode_j_per_tok", llm_decode_j_per_tok, "J/tok")
        if not llm_avg_power_w:
            print("[device] warning: no llm device samples were collected")
    _print_summary_footer()

    for idx, run in enumerate(llm_runs, start=1):
        csv_rows.append(
            {
                "type": "llm",
                "image_resolution": llm_resolution,
                "repeat_index": idx,
                "vision_encode_ms": None,
                "vision_fps": None,
                "llm_prefill_tokens": run.num_prefill,
                "llm_decode_tokens": run.num_decode,
                "llm_prefill_tps": run.prefill_tps,
                "llm_decode_tps": run.decode_tps,
                "llm_ttft_ms": run.prefill_latency * 1000.0,
                "llm_decode_ms": run.decode_duration * 1000.0,
                "llm_total_ms": run.total_time * 1000.0,
                "avg_power_w": run.avg_power_w,
                "p99_power_w": run.p99_power_w,
                "avg_utilization_pct": run.avg_utilization_pct,
                "p99_utilization_pct": run.p99_utilization_pct,
                "avg_memory_used_mb": run.avg_memory_used_mb,
                "p99_memory_used_mb": run.p99_memory_used_mb,
                "total_memory_mb": run.total_memory_mb,
                "avg_memory_used_pct": run.avg_memory_used_pct,
                "p99_memory_used_pct": run.p99_memory_used_pct,
                "total_energy_j": run.total_energy_j,
                "prefill_tok_per_j": run.prefill_tokens_per_j,
                "decode_tok_per_j": run.decode_tokens_per_j,
                "prefill_j_per_tok": run.prefill_j_per_token,
                "decode_j_per_tok": run.decode_j_per_token,
                "vision_img_per_j": None,
                "vision_j_per_img": None,
            }
        )

    if args.json:
        _write_json(
            args.json,
            {
                "task": args.task,
                "model": args.model,
                "prompt": args.prompt,
                "decode": args.decode,
                "vision_results": resolution_payloads,
                "llm_reference_resolution": llm_resolution,
                "llm_results": {
                    "repeat": args.repeat,
                    "runs": [asdict(r) for r in llm_runs],
                    "summary": {
                        "llm_prefill_tps": _summary(llm_prefill_tps),
                        "llm_decode_tps": _summary(llm_decode_tps),
                        "llm_ttft_ms": _summary(llm_ttft_ms),
                        "llm_decode_duration_ms": _summary(llm_decode_ms),
                        "llm_total_ms": _summary(llm_total_ms),
                        "avg_power_w": _summary(llm_avg_power_w),
                        "p99_power_w": _summary(llm_p99_power_w),
                        "avg_utilization_pct": _summary(llm_avg_utilization_pct),
                        "p99_utilization_pct": _summary(llm_p99_utilization_pct),
                        "avg_memory_used_mb": _summary(llm_avg_memory_used_mb),
                        "p99_memory_used_mb": _summary(llm_p99_memory_used_mb),
                        "total_memory_mb": _summary(llm_total_memory_mb),
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
        )
        print(f"wrote: {args.json}")

    if args.csv:
        _write_csv(args.csv, csv_rows)
        print(f"wrote: {args.csv}")

    return 0


def add_tps_parser(
    subparsers: argparse._SubParsersAction[HfArgumentParser],
) -> None:
    parser = subparsers.add_parser("tps", help="Measure/sweep tokens-per-second")
    tps_sub = parser.add_subparsers(dest="tps_cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--task", default="text-generation", help="transformers pipeline task"
        )
        p.add_argument(
            "--model",
            required=True,
            help="model id or local path (e.g., mobilint/Llama-3.2-3B-Instruct)",
        )
        p.add_argument(
            "--tokenizer",
            default=None,
            help="tokenizer id or local path (defaults to model)",
        )
        p.add_argument(
            "--device", default="cpu", help="device for pipeline (e.g., cpu, cuda:0)"
        )
        p.add_argument(
            "--revision",
            default=None,
            help="model revision (e.g., W8)",
        )
        p.add_argument(
            "--embedding-weight",
            default=None,
            help="path to custom embedding weights",
        )
        p.add_argument(
            "--mxq-path",
            default=None,
            help="override mxq_path for pipeline loading",
        )
        p.add_argument(
            "--core-mode",
            default=None,
            help="NPU core mode (single, multi, global4, global8)",
        )
        p.add_argument(
            "--target-cores",
            type=_parse_target_cores,
            default=None,
            help='Target cores (e.g., "0:0;0:1;0:2;0:3")',
        )
        p.add_argument(
            "--target-clusters",
            type=_parse_target_clusters,
            default=None,
            help='Target clusters (e.g., "0;1")',
        )
        p.add_argument(
            "--device-map", default=None, help="transformers device_map (optional)"
        )
        p.add_argument(
            "--dtype", default=None, help="dtype (e.g., auto, float16, bfloat16)"
        )
        p.add_argument(
            "--trust-remote-code",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="pass trust_remote_code to transformers",
        )
        p.add_argument(
            "--repeat", type=_parse_positive_int, default=1, help="number of repeated runs"
        )
        p.add_argument(
            "--warmup",
            type=_parse_positive_int,
            default=1,
            help="number of warmup runs before measured runs",
        )
        p.add_argument(
            "--trace",
            default=None,
            help="write qbruntime trace to the given JSON path (first run only)",
        )
        p.add_argument(
            "--chunk-size",
            type=_parse_positive_int_optional,
            default=None,
            help="optional chunk_size forwarded to model.generate/model.forward (default: None)",
        )
        _add_device_tracking_args(p)

    p_measure = tps_sub.add_parser("measure", help="Single TPS measurement")
    add_common(p_measure)
    p_measure.add_argument("--prefill", type=_parse_positive_int, default=512, help="input token count")
    p_measure.add_argument(
        "--decode", type=_parse_positive_int, default=128, help="new tokens to generate"
    )
    p_measure.add_argument("--json", default=None, help="write result as JSON")
    p_measure.set_defaults(_handler=_cmd_measure)

    p_sweep = tps_sub.add_parser("sweep", help="Prefill/decode TPS sweep")
    add_common(p_sweep)
    p_sweep.add_argument(
        "--prefill-range",
        type=_parse_range,
        default=(128, 512, 128),
        help="prefill sweep range (start:end:step)",
    )
    p_sweep.add_argument(
        "--decode-range",
        type=_parse_range,
        default=(128, 512, 128),
        help="decode sweep range (start:end:step)",
    )
    p_sweep.add_argument(
        "--fixed-decode",
        type=_parse_positive_int,
        default=10,
        help="fixed decode length for prefill sweep",
    )
    p_sweep.add_argument(
        "--fixed-prefill",
        type=_parse_positive_int,
        default=128,
        help="fixed prefill length for decode sweep",
    )
    p_sweep.add_argument("--plot", default="tps_benchmark.png", help="write PNG plot")
    p_sweep.add_argument(
        "--no-plot",
        dest="plot",
        action="store_const",
        const=None,
        help="disable plot output",
    )
    p_sweep.add_argument("--json", default=None, help="write sweep result as JSON")
    p_sweep.add_argument("--csv", default=None, help="write sweep rows as CSV")
    p_sweep.set_defaults(_handler=_cmd_sweep)

    p_vlm = tps_sub.add_parser("vlm-sweep", help="Synthetic VLM benchmark by image resolution")
    add_common(p_vlm)
    p_vlm.set_defaults(task="image-text-to-text")
    p_vlm.add_argument(
        "--image-resolutions",
        type=_parse_int_list,
        default=[224, 384, 512, 768],
        help="comma-separated image resolutions (e.g., 224,384,512)",
    )
    p_vlm.add_argument(
        "--decode",
        type=_parse_positive_int,
        default=128,
        help="decode tokens for LLM phase",
    )
    p_vlm.add_argument(
        "--llm-resolution",
        type=_parse_positive_int_optional,
        default=None,
        help="reference resolution used for LLM-only benchmark (default: first image resolution)",
    )
    p_vlm.add_argument(
        "--prompt",
        default="Describe the image in one sentence.",
        help="fixed prompt used for synthetic VLM input",
    )
    p_vlm.add_argument("--json", default=None, help="write VLM results as JSON")
    p_vlm.add_argument("--csv", default=None, help="write VLM rows as CSV")
    p_vlm.set_defaults(_handler=_cmd_vlm_sweep)

