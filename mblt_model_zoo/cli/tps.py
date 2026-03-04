from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Iterable, Sequence, Tuple, Union

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

    if dtype:
        try:
            pipeline_kwargs["dtype"] = dtype
            return hf_pipeline(**pipeline_kwargs)
        except TypeError:
            pipeline_kwargs.pop("dtype", None)
            pipeline_kwargs["torch_dtype"] = dtype
            return hf_pipeline(**pipeline_kwargs)

    return hf_pipeline(**pipeline_kwargs)


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
    for _ in range(args.warmup):
        measurer.measure(
            num_prefill=args.prefill,
            num_decode=args.decode,
            trace_path=None,
        )
    runs = [
        measurer.measure(
            num_prefill=args.prefill,
            num_decode=args.decode,
            trace_path=args.trace if i == 0 else None,
        )
        for i in range(args.repeat)
    ]

    prefill_tps = [r.prefill_tps for r in runs]
    decode_tps = [r.decode_tps for r in runs]
    ttft_ms = [r.prefill_latency * 1000.0 for r in runs]
    decode_ms = [r.decode_duration * 1000.0 for r in runs]
    total_ms = [r.total_time * 1000.0 for r in runs]

    print(f"warmup: {args.warmup}")
    print(f"runs: {args.repeat}")
    print(f"prefill tokens: {runs[0].num_prefill} | decode tokens: {runs[0].num_decode}")
    _print_summary_header()
    _print_summary("prefill_tps", prefill_tps, "tok/s")
    _print_summary("decode_tps", decode_tps, "tok/s")
    _print_summary("ttft", ttft_ms, "ms")
    _print_summary("decode_duration", decode_ms, "ms")
    _print_summary("total", total_ms, "ms")
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
    for _ in range(args.warmup):
        measurer.measure(
            num_prefill=args.fixed_prefill,
            num_decode=args.fixed_decode,
            trace_path=None,
        )
    runs = []
    for i in range(args.repeat):
        runs.append(
            measurer.measure_full(
                prefill_range=args.prefill_range,
                decode_range=args.decode_range,
                fixed_decode_len=args.fixed_decode,
                fixed_prefill_len=args.fixed_prefill,
                trace_path=args.trace if i == 0 else None,
            )
        )

    result = _aggregate_sweep_results(runs)

    prefill_last = [r.prefill_sweep.tps_values[-1] for r in runs if r.prefill_sweep.tps_values]
    decode_last = [r.decode_sweep.tps_values[-1] for r in runs if r.decode_sweep.tps_values]
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.repeat}")
    _print_summary_header()
    if prefill_last:
        _print_summary("prefill_tps(last_point)", prefill_last, "tok/s")
    if decode_last:
        _print_summary("decode_tps(last_point)", decode_last, "tok/s")
    _print_summary_footer()

    if args.json:
        payload = {
            "repeat": args.repeat,
            "aggregate": asdict(result),
            "runs": [asdict(r) for r in runs],
            "summary": {
                "prefill_tps_last": _summary(prefill_last),
                "decode_tps_last": _summary(decode_last),
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
        vision_runs = measurer.measure_vision(
            image_resolution=resolution,
            repeat=args.repeat,
            prompt=args.prompt,
            show_progress=True,
        )

        vision_ms = [lat * 1000.0 for lat, _ in vision_runs]
        vision_fps = [fps for _, fps in vision_runs]

        print(f"\nresolution={resolution} warmup={args.warmup} runs={args.repeat}")
        _print_summary_header()
        _print_summary("vision_encode", vision_ms, "ms")
        _print_summary("vision_fps", vision_fps, "fps")
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
    llm_runs = measurer.measure_llm(
        image_resolution=llm_resolution,
        num_decode=args.decode,
        repeat=args.repeat,
        prompt=args.prompt,
        show_progress=True,
    )

    llm_prefill_tps = [r.prefill_tps for r in llm_runs]
    llm_decode_tps = [r.decode_tps for r in llm_runs]
    llm_ttft_ms = [r.prefill_latency * 1000.0 for r in llm_runs]
    llm_decode_ms = [r.decode_duration * 1000.0 for r in llm_runs]
    llm_total_ms = [r.total_time * 1000.0 for r in llm_runs]

    print(
        f"\nllm_reference_resolution={llm_resolution} warmup={args.warmup} runs={args.repeat}"
    )
    _print_summary_header()
    _print_summary("llm_prefill_tps", llm_prefill_tps, "tok/s")
    _print_summary("llm_decode_tps", llm_decode_tps, "tok/s")
    _print_summary("llm_ttft", llm_ttft_ms, "ms")
    _print_summary("llm_decode_duration", llm_decode_ms, "ms")
    _print_summary("llm_total", llm_total_ms, "ms")
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
