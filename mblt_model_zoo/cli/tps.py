from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Iterable, Sequence, Tuple

from transformers import HfArgumentParser


def _parse_range(spec: str) -> Tuple[int, int, int]:
    """
    Parse a sweep range spec.

    Supported formats:
      - "start:end:step"  (e.g., "128:2048:128")
      - "start,end,step"  (e.g., "128,2048,128")
    """
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


def _parse_target_cores(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    text = spec.strip()
    if not text:
        return None
    return [item.strip() for item in text.split(";") if item.strip()]


def _parse_target_clusters(spec: str | None) -> list[int] | None:
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
    tokenizer: str | None,
    device: str,
    trust_remote_code: bool,
    dtype: str | None,
    device_map: str | None,
    revision: str | None,
    embedding_weight: str | None,
    mxq_path: str | None,
    core_mode: str | None,
    target_cores: list[str] | None,
    target_clusters: list[int] | None,
) -> Any:
    _require_transformers_deps()
    from mblt_model_zoo.transformers.utils.auto import pipeline as hf_pipeline

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
    for x, tps, t in zip(
        result.prefill_sweep.x_values,
        result.prefill_sweep.tps_values,
        result.prefill_sweep.time_values,
    ):
        yield {"phase": "prefill", "tokens": x, "tps": tps, "time_s": t}
    for x, tps, t in zip(
        result.decode_sweep.x_values,
        result.decode_sweep.tps_values,
        result.decode_sweep.time_values,
    ):
        yield {"phase": "decode", "tokens": x, "tps": tps, "time_s": t}


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fieldnames = ["phase", "tokens", "tps", "time_s"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    from mblt_model_zoo.transformers.utils.benchmark_utils import TPSMeasurer

    measurer = TPSMeasurer(pipeline)
    for _ in range(args.warmup):
        measurer.measure(num_prefill=args.prefill, num_decode=args.decode)

    res = measurer.measure(
        num_prefill=args.prefill,
        num_decode=args.decode,
        trace_path=args.trace,
    )

    print(
        f"prefill: {res.num_prefill} tokens | {res.prefill_tps:.2f} tok/s | TTFT {res.prefill_latency:.4f}s"
    )
    print(
        f"decode:  {res.num_decode} tokens | {res.decode_tps:.2f} tok/s | duration {res.decode_duration:.4f}s"
    )
    print(f"total:   {res.total_time:.4f}s")

    if args.json:
        _write_json(args.json, asdict(res))
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

    from mblt_model_zoo.transformers.utils.benchmark_utils import TPSMeasurer

    measurer = TPSMeasurer(pipeline)
    for _ in range(args.warmup):
        measurer.measure(num_prefill=args.fixed_prefill, num_decode=args.fixed_decode)

    result = measurer.measure_full(
        prefill_range=args.prefill_range,
        decode_range=args.decode_range,
        fixed_decode_len=args.fixed_decode,
        fixed_prefill_len=args.fixed_prefill,
        trace_path=args.trace,
    )

    if args.json:
        _write_json(args.json, asdict(result))
        print(f"wrote: {args.json}")

    if args.csv:
        rows = list(_iter_rows_for_csv(result))
        _write_csv(args.csv, rows)
        print(f"wrote: {args.csv}")

    if args.plot:
        measurer.plot_and_save(result, save_path=args.plot)

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
            "--warmup", type=int, default=1, help="warmup runs before measuring"
        )
        p.add_argument(
            "--trace",
            default=None,
            help="write qbruntime trace to the given JSON path",
        )

    p_measure = tps_sub.add_parser("measure", help="Single TPS measurement")
    add_common(p_measure)
    p_measure.add_argument("--prefill", type=int, default=512, help="input token count")
    p_measure.add_argument(
        "--decode", type=int, default=128, help="new tokens to generate"
    )
    p_measure.add_argument("--json", default=None, help="write single result as JSON")
    p_measure.set_defaults(_handler=_cmd_measure)

    p_sweep = tps_sub.add_parser("sweep", help="Prefill/decode TPS sweep")
    add_common(p_sweep)
    p_sweep.add_argument(
        "--prefill-range",
        type=_parse_range,
        default=(128, 2048, 128),
        help="prefill sweep range (start:end:step)",
    )
    p_sweep.add_argument(
        "--decode-range",
        type=_parse_range,
        default=(128, 1024, 128),
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
