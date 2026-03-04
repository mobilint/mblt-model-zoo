from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

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
    try:
        value = int(spec)
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected integer") from e
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _parse_int_list_optional(spec: Union[str, None]) -> Union[list[int], None]:
    if spec is None:
        return None
    text = spec.strip()
    if not text:
        return None
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        return None
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("power-gpu-id values must be >= 0")
    return values


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
                "This happens before power tracking starts and is a host GPU driver/runtime issue.\n"
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


def _infer_gpu_ids(device: str, power_gpu_id: Optional[list[int]]) -> Optional[Union[int, list[int]]]:
    if power_gpu_id is not None:
        return power_gpu_id[0] if len(power_gpu_id) == 1 else power_gpu_id
    text = (device or "").strip().lower()
    if text.startswith("cuda:"):
        try:
            gpu_id = int(text.split(":", 1)[1])
            return gpu_id
        except ValueError:
            return None
    return None


def _build_power_tracker(args: argparse.Namespace, pipeline: Any):
    if not args.power:
        return None

    def _has_npu_backend(obj: Any, depth: int = 0, seen: Optional[set[int]] = None) -> bool:
        if obj is None:
            return False
        if seen is None:
            seen = set()
        oid = id(obj)
        if oid in seen:
            return False
        seen.add(oid)
        if hasattr(obj, "npu_backend"):
            return True
        if depth >= 2:
            return False
        for name in (
            "model",
            "language_model",
            "vision_model",
            "text_model",
            "encoder",
            "decoder",
        ):
            child = getattr(obj, name, None)
            if child is not None and _has_npu_backend(child, depth + 1, seen):
                return True
        return False

    is_mobilint_model = False
    try:
        from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin

        is_mobilint_model = isinstance(pipeline.model, MobilintModelMixin) or _has_npu_backend(pipeline.model)
    except Exception:
        is_mobilint_model = _has_npu_backend(getattr(pipeline, "model", None))

    backend = args.power_device
    if backend == "auto":
        if is_mobilint_model:
            backend = "npu"
        else:
            device_text = (args.device or "").strip().lower()
            backend = "gpu" if device_text.startswith("cuda") else "none"

    if backend == "none":
        return None

    if backend == "npu":
        from mblt_model_zoo.utils.power_tracker_npu import NPUPowerTracker
        try:
            return NPUPowerTracker(interval=args.power_interval)
        except Exception as e:
            print(f"[power] failed to initialize NPU tracker: {e}", file=sys.stderr)
            return None

    if backend == "gpu":
        from mblt_model_zoo.utils.power_tracker_gpu import GPUPowerTracker

        gpu_id = _infer_gpu_ids(args.device, args.power_gpu_id)
        try:
            return GPUPowerTracker(interval=args.power_interval, gpu_id=gpu_id)
        except Exception as e:
            print(f"[power] failed to initialize GPU tracker: {e}", file=sys.stderr)
            return None

    return None


def _avg_power_in_window(
    trace: Sequence[tuple[float, float]],
    start_t: float,
    end_t: float,
    default_power: Optional[float],
) -> Optional[float]:
    if end_t <= start_t:
        return default_power
    vals = [p for ts, p in trace if start_t <= ts <= end_t]
    if vals:
        return float(sum(vals) / len(vals))
    return default_power


def _print_power_status(args: argparse.Namespace, tracker: Any) -> None:
    if not args.power:
        print("[power] disabled by --no-power")
        return
    if tracker is None:
        print("[power] enabled but no compatible tracker initialized (auto detection fallback)")
        return
    print(f"[power] enabled with {tracker.__class__.__name__} (interval={args.power_interval}s)")


def _safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def _enrich_single_run_power(
    run: Any,
    power_metric: dict[str, Any],
    power_trace: Sequence[tuple[float, float]],
    run_start_t: float,
) -> None:
    avg_power = power_metric.get("avg_power_w")
    p99_power = power_metric.get("p99_power_w")
    if avg_power is None:
        return

    prefill_start = run_start_t
    prefill_end = run_start_t + run.prefill_latency
    decode_start = prefill_end
    decode_end = decode_start + run.decode_duration

    prefill_avg_power = _avg_power_in_window(power_trace, prefill_start, prefill_end, avg_power)
    decode_avg_power = _avg_power_in_window(power_trace, decode_start, decode_end, avg_power)

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
    run.p99_power_w = float(p99_power) if p99_power is not None else None
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
    tracker = _build_power_tracker(args, pipeline)
    _print_power_status(args, tracker)
    for _ in tqdm(range(args.warmup), desc="warmup", leave=False):
        measurer.measure(
            num_prefill=args.prefill,
            num_decode=args.decode,
            trace_path=None,
        )
    runs = []
    for i in tqdm(range(args.repeat), desc="measure runs", leave=False):
        run_start_t = time.time()
        if tracker is not None:
            tracker.start()
        try:
            run = measurer.measure(
                num_prefill=args.prefill,
                num_decode=args.decode,
                trace_path=args.trace if i == 0 else None,
            )
        finally:
            if tracker is not None:
                tracker.stop()
        if tracker is not None:
            _enrich_single_run_power(
                run=run,
                power_metric=tracker.get_power_metric(),
                power_trace=tracker.get_power_trace(),
                run_start_t=run_start_t,
            )
        runs.append(run)

    prefill_tps = [r.prefill_tps for r in runs]
    decode_tps = [r.decode_tps for r in runs]
    ttft_ms = [r.prefill_latency * 1000.0 for r in runs]
    decode_ms = [r.decode_duration * 1000.0 for r in runs]
    total_ms = [r.total_time * 1000.0 for r in runs]
    avg_power_w = [r.avg_power_w for r in runs if r.avg_power_w is not None]
    p99_power_w = [r.p99_power_w for r in runs if r.p99_power_w is not None]
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
    if args.power:
        _print_summary("avg_power", avg_power_w, "W")
        _print_summary("p99_power", p99_power_w, "W")
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
                "total_energy_j": _summary(total_energy_j),
                "prefill_tok_per_j": _summary(prefill_tok_per_j),
                "decode_tok_per_j_dfi": _summary(decode_tok_per_j),
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
    tracker = _build_power_tracker(args, pipeline)
    _print_power_status(args, tracker)
    for _ in tqdm(range(args.warmup), desc="warmup", leave=False):
        measurer.measure(
            num_prefill=args.fixed_prefill,
            num_decode=args.fixed_decode,
            trace_path=None,
        )
    runs = []
    run_avg_power: list[float] = []
    run_p99_power: list[float] = []
    run_total_energy: list[float] = []
    for i in tqdm(range(args.repeat), desc="sweep runs", leave=False):
        run_start_t = time.time()
        if tracker is not None:
            tracker.start()
        try:
            runs.append(
                measurer.measure_full(
                    prefill_range=args.prefill_range,
                    decode_range=args.decode_range,
                    fixed_decode_len=args.fixed_decode,
                    fixed_prefill_len=args.fixed_prefill,
                    trace_path=args.trace if i == 0 else None,
                    show_progress=True,
                )
            )
        finally:
            if tracker is not None:
                tracker.stop()
        if tracker is not None:
            metric = tracker.get_power_metric()
            avg_power = metric.get("avg_power_w")
            p99_power = metric.get("p99_power_w")
            if avg_power is not None:
                run_avg_power.append(float(avg_power))
            if p99_power is not None:
                run_p99_power.append(float(p99_power))
            elapsed = time.time() - run_start_t
            if avg_power is not None:
                run_total_energy.append(float(avg_power) * elapsed)

    result = _aggregate_sweep_results(runs)

    prefill_last = [r.prefill_sweep.tps_values[-1] for r in runs if r.prefill_sweep.tps_values]
    decode_last = [r.decode_sweep.tps_values[-1] for r in runs if r.decode_sweep.tps_values]
    prefill_last_tpj: list[float] = []
    decode_last_tpj: list[float] = []
    prefill_last_jpt: list[float] = []
    decode_last_jpt: list[float] = []
    for i, p_tps in enumerate(prefill_last):
        if i < len(run_avg_power) and run_avg_power[i] > 0:
            tpj = p_tps / run_avg_power[i]
            prefill_last_tpj.append(tpj)
            prefill_last_jpt.append(1.0 / tpj if tpj > 0 else 0.0)
    for i, d_tps in enumerate(decode_last):
        if i < len(run_avg_power) and run_avg_power[i] > 0:
            tpj = d_tps / run_avg_power[i]
            decode_last_tpj.append(tpj)
            decode_last_jpt.append(1.0 / tpj if tpj > 0 else 0.0)
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.repeat}")
    _print_summary_header()
    if prefill_last:
        _print_summary("prefill_tps(last_point)", prefill_last, "tok/s")
    if decode_last:
        _print_summary("decode_tps(last_point)", decode_last, "tok/s")
    if args.power:
        _print_summary("avg_power", run_avg_power, "W")
        _print_summary("p99_power", run_p99_power, "W")
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
                "total_energy_j": _summary(run_total_energy),
                "prefill_tok_per_j_last": _summary(prefill_last_tpj),
                "decode_tok_per_j_last_dfi": _summary(decode_last_tpj),
                "prefill_j_per_tok_last": _summary(prefill_last_jpt),
                "decode_j_per_tok_last": _summary(decode_last_jpt),
            },
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
    tracker = _build_power_tracker(args, pipeline)
    _print_power_status(args, tracker)

    resolution_payloads = []
    csv_rows: list[dict[str, Any]] = []
    for resolution in tqdm(args.image_resolutions, desc="vision resolutions"):
        for _ in tqdm(range(args.warmup), desc=f"warmup@{resolution}", leave=False):
            measurer.measure_vision(
                image_resolution=resolution,
                repeat=1,
                prompt=args.prompt,
                show_progress=False,
            )
        vision_runs = []
        vision_power_avg = []
        vision_power_p99 = []
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
                metric = tracker.get_power_metric()
                avg_power = metric.get("avg_power_w")
                p99_power = metric.get("p99_power_w")
                if avg_power is not None:
                    avg_power_f = float(avg_power)
                    energy = avg_power_f * float(single[0])
                    vision_power_avg.append(avg_power_f)
                    vision_energy_j.append(energy)
                    vision_img_per_j.append(1.0 / energy if energy > 0 else 0.0)
                    vision_j_per_img.append(energy)
                if p99_power is not None:
                    vision_power_p99.append(float(p99_power))

        vision_ms = [lat * 1000.0 for lat, _ in vision_runs]
        vision_fps = [fps for _, fps in vision_runs]

        print(f"\nresolution={resolution} warmup={args.warmup} runs={args.repeat}")
        _print_summary_header()
        _print_summary("vision_encode", vision_ms, "ms")
        _print_summary("vision_fps", vision_fps, "fps")
        if args.power:
            _print_summary("vision_avg_power", vision_power_avg, "W")
            _print_summary("vision_p99_power", vision_power_p99, "W")
            _print_summary("vision_energy", vision_energy_j, "J")
            _print_summary("vision_img_per_j", vision_img_per_j, "img/J")
            _print_summary("vision_j_per_img", vision_j_per_img, "J/img")
            if not vision_power_avg:
                print("[power] warning: no vision power samples were collected for this resolution")
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
                    "total_energy_j": vision_energy_j[idx - 1] if idx - 1 < len(vision_energy_j) else None,
                    "prefill_tok_per_j": None,
                    "decode_tok_per_j_dfi": None,
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
    for _ in tqdm(range(args.warmup), desc=f"llm warmup@{llm_resolution}", leave=False):
        measurer.measure_llm(
            image_resolution=llm_resolution,
            num_decode=args.decode,
            repeat=1,
            prompt=args.prompt,
            show_progress=False,
        )
    llm_runs = []
    for _ in tqdm(range(args.repeat), desc=f"llm@{llm_resolution}", leave=False):
        run_start_t = time.time()
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
            _enrich_single_run_power(
                run=run,
                power_metric=tracker.get_power_metric(),
                power_trace=tracker.get_power_trace(),
                run_start_t=run_start_t,
            )
        llm_runs.append(run)

    llm_prefill_tps = [r.prefill_tps for r in llm_runs]
    llm_decode_tps = [r.decode_tps for r in llm_runs]
    llm_ttft_ms = [r.prefill_latency * 1000.0 for r in llm_runs]
    llm_decode_ms = [r.decode_duration * 1000.0 for r in llm_runs]
    llm_total_ms = [r.total_time * 1000.0 for r in llm_runs]
    llm_avg_power_w = [r.avg_power_w for r in llm_runs if r.avg_power_w is not None]
    llm_p99_power_w = [r.p99_power_w for r in llm_runs if r.p99_power_w is not None]
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
    if args.power:
        _print_summary("llm_avg_power", llm_avg_power_w, "W")
        _print_summary("llm_p99_power", llm_p99_power_w, "W")
        _print_summary("llm_total_energy", llm_total_energy_j, "J")
        _print_summary("llm_prefill_tok_per_j", llm_prefill_tok_per_j, "tok/J")
        _print_summary("llm_decode_tok_per_j", llm_decode_tok_per_j, "tok/J")
        _print_summary("llm_prefill_j_per_tok", llm_prefill_j_per_tok, "J/tok")
        _print_summary("llm_decode_j_per_tok", llm_decode_j_per_tok, "J/tok")
        if not llm_avg_power_w:
            print("[power] warning: no llm power samples were collected")
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
                "total_energy_j": run.total_energy_j,
                "prefill_tok_per_j": run.prefill_tokens_per_j,
                "decode_tok_per_j_dfi": run.decode_tokens_per_j,
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
                        "total_energy_j": _summary(llm_total_energy_j),
                        "prefill_tok_per_j": _summary(llm_prefill_tok_per_j),
                        "decode_tok_per_j_dfi": _summary(llm_decode_tok_per_j),
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
            "--power",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="enable power tracking (default: on, disable via --no-power)",
        )
        p.add_argument(
            "--power-device",
            choices=["auto", "gpu", "npu"],
            default="auto",
            help="power backend selection (default: auto)",
        )
        p.add_argument(
            "--power-interval",
            type=float,
            default=0.2,
            help="power sampling interval in seconds",
        )
        p.add_argument(
            "--power-gpu-id",
            type=_parse_int_list_optional,
            default=None,
            help="comma-separated GPU ids for power tracking (e.g., 0,1)",
        )

    p_measure = tps_sub.add_parser("measure", help="Single TPS measurement")
    add_common(p_measure)
    p_measure.add_argument("--prefill", type=int, default=512, help="input token count")
    p_measure.add_argument(
        "--decode", type=int, default=128, help="new tokens to generate"
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
        type=int,
        default=10,
        help="fixed decode length for prefill sweep",
    )
    p_sweep.add_argument(
        "--fixed-prefill",
        type=int,
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
        type=int,
        default=128,
        help="decode tokens for LLM phase",
    )
    p_vlm.add_argument(
        "--llm-resolution",
        type=int,
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
