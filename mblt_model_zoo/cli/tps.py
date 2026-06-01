from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import torch
from tqdm.auto import tqdm

from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    CORE_MODE_CHOICES as _CORE_MODE_CHOICES,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    add_device_tracking_args as _add_device_tracking_args,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    apply_core_mode_model_kwargs as _apply_core_mode_model_kwargs_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    build_device_tracker as _build_device_tracker_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    build_phase_trackers as _build_phase_trackers_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    extract_device_metric as _extract_device_metric_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    extract_device_time_series as _extract_device_time_series_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    parse_positive_int as _parse_positive_int_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    parse_positive_int_optional as _parse_positive_int_optional_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    print_device_status as _print_device_status_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    resolve_default_device as _resolve_default_device_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    resolve_default_device_backend as _resolve_default_device_backend_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    stop_tracker_safe as _stop_tracker_safe_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    weighted_two as _weighted_two_common,
)

_SWEEP_WARMUP_PREFILL = 128
_SWEEP_WARMUP_DECODE = 32


@dataclass(frozen=True)
class Eagle3PipelineOptions:
    """Bundle EAGLE-3-specific pipeline options."""

    base_embedding_path: str | None = None
    draft_embedding_path: str | None = None
    base_mxq_path: str | None = None
    draft_mxq_path: str | None = None
    fc_mxq_path: str | None = None
    base_core_mode: str | None = None
    draft_core_mode: str | None = None
    fc_core_mode: str | None = None
    base_target_cores: list[str] | None = None
    draft_target_cores: list[str] | None = None
    fc_target_cores: list[str] | None = None
    base_target_clusters: list[int] | None = None
    draft_target_clusters: list[int] | None = None
    fc_target_clusters: list[int] | None = None


def _warn_eagle3_override(
    global_name: str,
    prefixed_name: str,
    global_value: Any,
    prefixed_value: Any,
    *,
    stacklevel: int = 2,
) -> None:
    """Warn when a prefixed EAGLE-3 option overrides the corresponding global option."""
    if global_value is None or prefixed_value is None:
        return
    if global_value == prefixed_value:
        return
    warnings.warn(
        (
            f"Conflicting options detected: `{global_name}` and `{prefixed_name}`. "
            f"Using `{prefixed_name}` value because EAGLE-3 prefixed options take precedence over global options."
        ),
        UserWarning,
        stacklevel=stacklevel,
    )


def _warn_eagle3_applied_options_summary(model_kwargs: dict[str, Any]) -> None:
    """Print once with the final EAGLE-3 backend-prefixed options.

    This helps users understand the effective option set when shared and
    prefixed CLI options are mixed.
    """
    tracked_keys = [
        "base_core_mode",
        "draft_core_mode",
        "fc_core_mode",
        "base_target_cores",
        "draft_target_cores",
        "fc_target_cores",
        "base_target_clusters",
        "draft_target_clusters",
        "fc_target_clusters",
        "base_mxq_path",
        "draft_mxq_path",
        "fc_mxq_path",
    ]
    applied = {key: model_kwargs[key] for key in tracked_keys if key in model_kwargs}
    if not applied:
        return
    if os.environ.get("MBLT_EAGLE3_VERBOSE", "0") == "1":
        print(f"[Mobilint][EAGLE-3] Applied backend options: {applied}", file=sys.stderr)


def npu_latency_pct(total_latency: Optional[float], npu_latency: Optional[float]) -> Optional[float]:
    """Return the NPU latency percentage of total latency."""
    if total_latency is None or npu_latency is None:
        return None
    if total_latency <= 0:
        return None
    return (npu_latency / total_latency) * 100.0


def _parse_range(spec: str) -> Tuple[int, int, int]:
    text = spec.strip()
    sep = ":" if ":" in text else ("," if "," in text else None)
    if sep is None:
        raise argparse.ArgumentTypeError(f"invalid range '{spec}': expected 'start:end:step' or 'start,end,step'")
    parts = [p.strip() for p in text.split(sep)]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"invalid range '{spec}': expected 3 integers (start, end, step)")
    try:
        start, end, step = (int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid range '{spec}': start/end/step must be integers") from e
    if step <= 0:
        raise argparse.ArgumentTypeError(f"invalid range '{spec}': step must be > 0")
    if start <= 0 or end <= 0:
        raise argparse.ArgumentTypeError(f"invalid range '{spec}': start/end must be > 0")
    if start > end:
        raise argparse.ArgumentTypeError(f"invalid range '{spec}': start must be <= end")
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


def _is_vlm_task(task: str) -> bool:
    """Return whether a transformers pipeline task should use VLM TPS measurement."""
    return task in {"image-text-to-text", "image-to-text"}


def _apply_vlm_core_mode_model_kwargs(
    model_kwargs: dict[str, Any],
    core_mode: str | None,
    *,
    target_cores: list[str] | None = None,
    target_clusters: list[int] | None = None,
) -> dict[str, Any]:
    """Apply shared VLM core-mode kwargs to both vision and text sub-configs."""
    expanded: dict[str, Any] = {}
    _apply_core_mode_model_kwargs_common(
        expanded,
        core_mode,
        target_cores=target_cores,
        target_clusters=target_clusters,
    )
    for prefix in ("vision", "text"):
        for key, value in expanded.items():
            model_kwargs[f"{prefix}_{key}"] = value
    return model_kwargs


def _normalize_max_batch_size(value: Any) -> int | None:
    """Normalize a model config max batch size value.

    Args:
        value: Candidate value read from a model config.

    Returns:
        A positive batch size when the candidate can be converted to an integer, otherwise ``None``.
    """
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return None


def _candidate_max_batch_sizes(config: Any, *, task: str) -> Iterable[Any]:
    """Yield task-specific max batch size candidates from a model config.

    Args:
        config: Model config object that may expose top-level or nested max batch size attributes.
        task: Transformers pipeline task used to decide VLM-specific candidates.

    Yields:
        Candidate max batch size values in priority order.
    """
    yield getattr(config, "max_batch_size", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        yield getattr(text_config, "max_batch_size", None)
    if _is_vlm_task(task):
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            yield getattr(vision_config, "max_batch_size", None)


def _resolve_model_max_batch_size(pipeline: Any, *, task: str) -> int:
    """Resolve the automatic CLI batch size from a loaded pipeline.

    Args:
        pipeline: Loaded transformers pipeline.
        task: Transformers pipeline task.

    Returns:
        The first valid model max batch size candidate, or ``1`` when unavailable.
    """
    model = getattr(pipeline, "model", None)
    config = getattr(model, "config", None)
    if config is None:
        return 1
    for candidate in _candidate_max_batch_sizes(config, task=task):
        batch_size = _normalize_max_batch_size(candidate)
        if batch_size is not None:
            return batch_size
    return 1


def _resolve_cli_batch_size(args: argparse.Namespace, pipeline: Any) -> int:
    """Resolve the effective TPS measurement batch size.

    Args:
        args: Parsed CLI arguments.
        pipeline: Loaded transformers pipeline.

    Returns:
        Explicit CLI batch size when provided, otherwise the model config batch size fallback.
    """
    if args.batch_size is not None:
        return int(args.batch_size)
    return _resolve_model_max_batch_size(pipeline, task=args.task)


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
    eagle3_options: Eagle3PipelineOptions,
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
    if eagle3_options.base_embedding_path:
        model_kwargs["base_embedding_weight"] = eagle3_options.base_embedding_path
    if eagle3_options.draft_embedding_path:
        model_kwargs["draft_embedding_weight"] = eagle3_options.draft_embedding_path
    if mxq_path:
        model_kwargs["mxq_path"] = mxq_path
    if eagle3_options.base_mxq_path:
        model_kwargs["base_mxq_path"] = eagle3_options.base_mxq_path
    if eagle3_options.draft_mxq_path:
        model_kwargs["draft_mxq_path"] = eagle3_options.draft_mxq_path
    if eagle3_options.fc_mxq_path:
        model_kwargs["fc_mxq_path"] = eagle3_options.fc_mxq_path
    eagle3_prefix_requested = any(
        value is not None
        for value in (
            eagle3_options.base_embedding_path,
            eagle3_options.draft_embedding_path,
            eagle3_options.base_mxq_path,
            eagle3_options.draft_mxq_path,
            eagle3_options.fc_mxq_path,
            eagle3_options.base_core_mode,
            eagle3_options.draft_core_mode,
            eagle3_options.fc_core_mode,
            eagle3_options.base_target_cores,
            eagle3_options.draft_target_cores,
            eagle3_options.fc_target_cores,
            eagle3_options.base_target_clusters,
            eagle3_options.draft_target_clusters,
            eagle3_options.fc_target_clusters,
        )
    )
    if _is_vlm_task(task):
        model_kwargs = _apply_vlm_core_mode_model_kwargs(
            model_kwargs,
            core_mode,
            target_cores=target_cores,
            target_clusters=target_clusters,
        )
    elif eagle3_prefix_requested:
        _warn_eagle3_override("--core-mode", "--base-core-mode", core_mode, eagle3_options.base_core_mode)
        _warn_eagle3_override("--core-mode", "--draft-core-mode", core_mode, eagle3_options.draft_core_mode)
        _warn_eagle3_override("--core-mode", "--fc-core-mode", core_mode, eagle3_options.fc_core_mode)
        _warn_eagle3_override("--target-cores", "--base-target-cores", target_cores, eagle3_options.base_target_cores)
        _warn_eagle3_override("--target-cores", "--draft-target-cores", target_cores, eagle3_options.draft_target_cores)
        _warn_eagle3_override("--target-cores", "--fc-target-cores", target_cores, eagle3_options.fc_target_cores)
        _warn_eagle3_override(
            "--target-clusters",
            "--base-target-clusters",
            target_clusters,
            eagle3_options.base_target_clusters,
        )
        _warn_eagle3_override(
            "--target-clusters",
            "--draft-target-clusters",
            target_clusters,
            eagle3_options.draft_target_clusters,
        )
        _warn_eagle3_override(
            "--target-clusters",
            "--fc-target-clusters",
            target_clusters,
            eagle3_options.fc_target_clusters,
        )
        _warn_eagle3_override("--mxq-path", "--base-mxq-path", mxq_path, eagle3_options.base_mxq_path)
        _warn_eagle3_override("--mxq-path", "--draft-mxq-path", mxq_path, eagle3_options.draft_mxq_path)
        _warn_eagle3_override("--mxq-path", "--fc-mxq-path", mxq_path, eagle3_options.fc_mxq_path)

        def _coalesce(preferred: Any, fallback: Any) -> Any:
            return preferred if preferred is not None else fallback

        for prefix, prefix_core_mode, prefix_target_cores, prefix_target_clusters in (
            (
                "base",
                _coalesce(eagle3_options.base_core_mode, core_mode),
                _coalesce(eagle3_options.base_target_cores, target_cores),
                _coalesce(eagle3_options.base_target_clusters, target_clusters),
            ),
            (
                "draft",
                _coalesce(eagle3_options.draft_core_mode, core_mode),
                _coalesce(eagle3_options.draft_target_cores, target_cores),
                _coalesce(eagle3_options.draft_target_clusters, target_clusters),
            ),
            (
                "fc",
                _coalesce(eagle3_options.fc_core_mode, core_mode),
                _coalesce(eagle3_options.fc_target_cores, target_cores),
                _coalesce(eagle3_options.fc_target_clusters, target_clusters),
            ),
        ):
            model_kwargs = _apply_core_mode_model_kwargs_common(
                model_kwargs,
                prefix_core_mode,
                target_cores=prefix_target_cores,
                target_clusters=prefix_target_clusters,
                prefix=prefix,
            )
        _warn_eagle3_applied_options_summary(model_kwargs)
    else:
        model_kwargs = _apply_core_mode_model_kwargs_common(
            model_kwargs,
            core_mode,
            target_cores=target_cores,
            target_clusters=target_clusters,
        )
    if model_kwargs:
        pipeline_kwargs["model_kwargs"] = model_kwargs

    def _raise_cuda_nvml_hint(exc: Exception) -> None:
        msg = str(exc)
        if "nvmlInit_v2" in msg or "Can't initialize NVML" in msg:
            raise SystemExit(
                "CUDA/NVML initialization failed while creating the pipeline.\n"
                "This happens before device tracking starts and is a host GPU driver/runtime issue.\n"
                'Check: `nvidia-smi`, `python -c "import torch; print(torch.cuda.is_available())"`.\n'
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


def _extract_eagle3_pipeline_kwargs(args: argparse.Namespace) -> Eagle3PipelineOptions:
    """Return EAGLE-3-specific pipeline kwargs from parsed CLI arguments."""
    return Eagle3PipelineOptions(
        base_embedding_path=args.base_embedding_path,
        draft_embedding_path=args.draft_embedding_path,
        base_mxq_path=args.base_mxq_path,
        draft_mxq_path=args.draft_mxq_path,
        fc_mxq_path=args.fc_mxq_path,
        base_core_mode=args.base_core_mode,
        draft_core_mode=args.draft_core_mode,
        fc_core_mode=args.fc_core_mode,
        base_target_cores=args.base_target_cores,
        draft_target_cores=args.draft_target_cores,
        fc_target_cores=args.fc_target_cores,
        base_target_clusters=args.base_target_clusters,
        draft_target_clusters=args.draft_target_clusters,
        fc_target_clusters=args.fc_target_clusters,
    )


def _resolve_text_measure_inputs(
    args: argparse.Namespace,
    pipeline: Any,
) -> tuple[torch.Tensor | None, int, str | None]:
    """Resolve text-measure input ids/prefill length from CLI input-mode options."""

    def _tokenize_prompt_text(text: str) -> torch.Tensor:
        encoded = pipeline.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return encoded["input_ids"]

    selected_prompt_text: str | None = None
    input_mode = str(getattr(args, "input_mode", "random"))
    if input_mode == "random":
        return None, int(args.prefill), None

    if input_mode == "synthetic-text":
        if not args.prompt_text:
            raise ValueError("--input-mode synthetic-text requires --prompt-text.")
        selected_prompt_text = args.prompt_text
        input_ids = _tokenize_prompt_text(args.prompt_text)
        return input_ids, int(input_ids.shape[1]), selected_prompt_text

    if input_mode == "file":
        if not args.prompt_file:
            raise ValueError("--input-mode file requires --prompt-file.")
        with open(args.prompt_file, encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
        if not prompts:
            raise ValueError(f"No non-empty prompt found in file: {args.prompt_file}")

        strategy = str(getattr(args, "prompt_file_strategy", "first"))
        if strategy == "first":
            selected_prompt_text = prompts[0]
        elif strategy == "random":
            rng = random.Random(args.prompt_file_seed)
            selected_prompt_text = rng.choice(prompts)
        else:
            raise ValueError(f"Unsupported --prompt-file-strategy: {strategy}")

        input_ids = _tokenize_prompt_text(selected_prompt_text)
        return input_ids, int(input_ids.shape[1]), selected_prompt_text

    raise ValueError(f"Unsupported input mode: {input_mode}")


def _iter_rows_for_csv(result: Any) -> Iterable[dict[str, Any]]:
    def _optional_values(values: Sequence[Any], length: int) -> list[Any]:
        return [values[idx] if idx < len(values) else None for idx in range(length)]

    prefill_len = min(
        len(result.prefill_sweep.x_values),
        len(result.prefill_sweep.tps_values),
        len(result.prefill_sweep.time_values),
    )
    for x, tps, t, avg_total, avg_npu in zip(
        result.prefill_sweep.x_values[:prefill_len],
        result.prefill_sweep.tps_values[:prefill_len],
        result.prefill_sweep.time_values[:prefill_len],
        _optional_values(result.prefill_sweep.avg_total_token_latency_values, prefill_len),
        _optional_values(result.prefill_sweep.avg_npu_token_latency_values, prefill_len),
    ):
        yield {
            "phase": "prefill",
            "tokens": x,
            "tps": tps,
            "time_ms": t * 1000.0,
            "avg_total_token_latency_ms": avg_total * 1000.0 if avg_total is not None else None,
            "avg_npu_token_latency_ms": avg_npu * 1000.0 if avg_npu is not None else None,
            "avg_npu_token_latency_pct": npu_latency_pct(avg_total, avg_npu),
        }
    decode_len = min(
        len(result.decode_sweep.x_values),
        len(result.decode_sweep.tps_values),
        len(result.decode_sweep.time_values),
    )
    for x, tps, t, avg_total, avg_npu in zip(
        result.decode_sweep.x_values[:decode_len],
        result.decode_sweep.tps_values[:decode_len],
        result.decode_sweep.time_values[:decode_len],
        _optional_values(result.decode_sweep.avg_total_token_latency_values, decode_len),
        _optional_values(result.decode_sweep.avg_npu_token_latency_values, decode_len),
    ):
        yield {
            "phase": "decode",
            "tokens": x,
            "tps": tps,
            "time_ms": t * 1000.0,
            "avg_total_token_latency_ms": avg_total * 1000.0 if avg_total is not None else None,
            "avg_npu_token_latency_ms": avg_npu * 1000.0 if avg_npu is not None else None,
            "avg_npu_token_latency_pct": npu_latency_pct(avg_total, avg_npu),
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
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


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


def _extract_device_time_series(tracker: Any) -> dict[str, list[dict[str, float]]]:
    return _extract_device_time_series_common(tracker)


def _weighted_two(
    a: Optional[float],
    a_weight: float,
    b: Optional[float],
    b_weight: float,
) -> Optional[float]:
    return _weighted_two_common(a, a_weight, b, b_weight)


def _print_device_status(args: argparse.Namespace, tracker: Any) -> None:
    _print_device_status_common(args, tracker)


def _normalize_runtime_defaults(args: argparse.Namespace) -> None:
    args.device = _resolve_default_device_common(
        device=args.device,
        device_explicit=args.device is not None,
        model_id=args.model,
        mxq_path=args.mxq_path,
    )
    args.device_backend = _resolve_default_device_backend_common(
        device_backend=args.device_backend or "none",
        device_backend_explicit=args.device_backend is not None,
        model_id=args.model,
        mxq_path=args.mxq_path,
    )


def _safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def _enrich_single_run_device(
    run: Any,
    prefill_metric: dict[str, Optional[float]],
    decode_metric: dict[str, Optional[float]],
    batch_size: int = 1,
) -> None:
    """Attach device metrics to a benchmark run.

    Args:
        run: Single-run or aggregate benchmark result object to enrich.
        prefill_metric: Device metrics measured during the prefill phase.
        decode_metric: Device metrics measured during the decode phase.
        batch_size: Number of sequences measured in parallel.
    """
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
    run.prefill_avg_temperature_c = prefill_metric.get("avg_temperature_c")
    run.prefill_p99_temperature_c = prefill_metric.get("p99_temperature_c")
    run.decode_avg_temperature_c = decode_metric.get("avg_temperature_c")
    run.decode_p99_temperature_c = decode_metric.get("p99_temperature_c")
    run.prefill_avg_memory_used_mb = prefill_metric.get("avg_memory_used_mb")
    run.prefill_p99_memory_used_mb = prefill_metric.get("p99_memory_used_mb")
    run.decode_avg_memory_used_mb = decode_metric.get("avg_memory_used_mb")
    run.decode_p99_memory_used_mb = decode_metric.get("p99_memory_used_mb")
    run.prefill_avg_memory_used_pct = prefill_metric.get("avg_memory_used_pct")
    run.prefill_p99_memory_used_pct = prefill_metric.get("p99_memory_used_pct")
    run.decode_avg_memory_used_pct = decode_metric.get("avg_memory_used_pct")
    run.decode_p99_memory_used_pct = decode_metric.get("p99_memory_used_pct")

    fallback_prefill_t = 0.0
    if getattr(run, "prefill_sweep", None) and run.prefill_sweep.time_values:
        fallback_prefill_t = run.prefill_sweep.time_values[-1]
    fallback_decode_t = 0.0
    if getattr(run, "decode_sweep", None) and run.decode_sweep.time_values:
        fallback_decode_t = run.decode_sweep.time_values[-1]
    prefill_t = float(getattr(run, "prefill_latency", fallback_prefill_t))
    decode_t = float(getattr(run, "decode_duration", fallback_decode_t))
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
    run.avg_temperature_c = _weighted_two(
        prefill_metric.get("avg_temperature_c"),
        prefill_t,
        decode_metric.get("avg_temperature_c"),
        decode_t,
    )
    p_t_p99 = prefill_metric.get("p99_temperature_c")
    d_t_p99 = decode_metric.get("p99_temperature_c")
    run.p99_temperature_c = max([v for v in (p_t_p99, d_t_p99) if v is not None], default=None)
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

    prefill_energy = prefill_avg_power * prefill_t if prefill_avg_power is not None else None
    decode_energy = decode_avg_power * decode_t if decode_avg_power is not None else None
    total_energy = None
    if prefill_energy is not None and decode_energy is not None:
        total_energy = prefill_energy + decode_energy

    num_prefill = int(
        getattr(
            run,
            "num_prefill",
            run.prefill_sweep.x_values[-1] if getattr(run, "prefill_sweep", None) and run.prefill_sweep.x_values else 0,
        )
    )
    num_decode = int(
        getattr(
            run,
            "num_decode",
            run.decode_sweep.x_values[-1] if getattr(run, "decode_sweep", None) and run.decode_sweep.x_values else 0,
        )
    )
    batch_size = max(1, int(batch_size))
    total_prefill_tokens = num_prefill * batch_size
    total_decode_tokens = num_decode * batch_size
    total_tokens = total_prefill_tokens + total_decode_tokens

    run.avg_power_w = float(avg_power)
    run.total_energy_j = total_energy
    run.prefill_tokens_per_j = (
        _safe_div(float(total_prefill_tokens), prefill_energy) if prefill_energy is not None else None
    )
    run.prefill_j_per_token = (
        _safe_div(prefill_energy, float(total_prefill_tokens))
        if prefill_energy is not None and total_prefill_tokens > 0
        else None
    )
    run.decode_tokens_per_j = (
        _safe_div(float(total_decode_tokens), decode_energy) if decode_energy is not None else None
    )
    run.decode_j_per_token = (
        _safe_div(decode_energy, float(total_decode_tokens))
        if decode_energy is not None and total_decode_tokens > 0
        else None
    )
    run.total_tokens_per_j = _safe_div(float(total_tokens), total_energy) if total_energy is not None else None
    run.total_j_per_token = (
        _safe_div(total_energy, float(total_tokens)) if total_energy is not None and total_tokens > 0 else None
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

        def _get_optional(values: Sequence[Any], idx: int) -> Any:
            return values[idx] if idx < len(values) else None

        for idx in range(len(first.x_values)):
            tps_vals = []
            time_vals = []
            total_vals = []
            npu_vals = []
            for result in results:
                src = result.prefill_sweep if phase == "prefill" else result.decode_sweep
                tps_vals.append(src.tps_values[idx])
                time_vals.append(src.time_values[idx])
                total_vals.append(_get_optional(src.avg_total_token_latency_values, idx))
                npu_vals.append(_get_optional(src.avg_npu_token_latency_values, idx))
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
    """Dispatch a single TPS measurement to the text or VLM measurement path."""
    if _is_vlm_task(args.task):
        return _run_vlm_measure(args)
    return _run_text_measure(args)


def _run_text_measure(args: argparse.Namespace) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _normalize_runtime_defaults(args)
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
        eagle3_options=_extract_eagle3_pipeline_kwargs(args),
        target_cores=args.target_cores,
        target_clusters=args.target_clusters,
    )
    batch_size = _resolve_cli_batch_size(args, pipeline)

    from mblt_model_zoo.hf_transformers.utils.benchmark_utils import TPSMeasurer

    measurer = TPSMeasurer(pipeline)
    tracker_prefill, tracker_decode = _build_phase_trackers(args, pipeline)
    _print_device_status(args, tracker_prefill)

    measure_input_ids, measure_num_prefill, selected_prompt_text = _resolve_text_measure_inputs(args, pipeline)
    if measure_input_ids is not None:
        resolved_batch = int(measure_input_ids.shape[0])
        if resolved_batch == 1 and batch_size > 1:
            measure_input_ids = measure_input_ids.repeat(batch_size, 1)
        elif resolved_batch != batch_size:
            raise ValueError(
                "Resolved prompt input batch size does not match effective batch size: "
                f"input_ids batch={resolved_batch}, effective batch_size={batch_size}."
            )

    selected_prompt_sha256 = (
        hashlib.sha256(selected_prompt_text.encode("utf-8")).hexdigest() if selected_prompt_text is not None else None
    )

    for i in tqdm(range(args.warmup), desc="warmup runs", leave=False):
        measurer.measure(
            num_prefill=measure_num_prefill,
            num_decode=args.decode,
            input_ids=measure_input_ids,
            prefill_chunk_size=args.prefill_chunk_size,
            trace_path=None,
            show_progress=True,
            progress_desc=f"warmup generate {i + 1}/{args.warmup}",
            batch_size=batch_size,
        )
    runs = []
    run_phase_device_time_series: list[dict[str, dict[str, list[dict[str, float]]]]] = []
    for i in tqdm(range(args.repeat), desc="measure runs", leave=False):
        prefill_metric: dict[str, Optional[float]] = {}
        decode_metric: dict[str, Optional[float]] = {}
        try:
            run = measurer.measure(
                num_prefill=measure_num_prefill,
                num_decode=args.decode,
                input_ids=measure_input_ids,
                prefill_chunk_size=args.prefill_chunk_size,
                trace_path=args.trace if i == 0 else None,
                show_progress=True,
                progress_desc=f"measure generate {i + 1}/{args.repeat}",
                on_prefill_start=((lambda: tracker_prefill.start()) if tracker_prefill is not None else None),
                on_prefill_end=((lambda: tracker_prefill.stop()) if tracker_prefill is not None else None),
                on_decode_start=((lambda: tracker_decode.start()) if tracker_decode is not None else None),
                on_decode_end=((lambda: tracker_decode.stop()) if tracker_decode is not None else None),
                batch_size=batch_size,
            )
        finally:
            _stop_tracker_safe(tracker_prefill)
            _stop_tracker_safe(tracker_decode)
        if tracker_prefill is not None and tracker_decode is not None:
            prefill_metric = _extract_device_metric(tracker_prefill)
            decode_metric = _extract_device_metric(tracker_decode)
            run_phase_device_time_series.append(
                {
                    "prefill": _extract_device_time_series(tracker_prefill),
                    "decode": _extract_device_time_series(tracker_decode),
                }
            )
            _enrich_single_run_device(
                run=run,
                prefill_metric=prefill_metric,
                decode_metric=decode_metric,
                batch_size=batch_size,
            )
        runs.append(run)

    prefill_tps = [r.prefill_tps for r in runs]
    decode_tps = [r.decode_tps for r in runs]
    ttft_ms = [r.prefill_latency * 1000.0 for r in runs]
    decode_ms = [r.decode_duration * 1000.0 for r in runs]
    total_ms = [r.total_time * 1000.0 for r in runs]
    prefill_npu_latency_pct = [pct for r in runs if (pct := r.prefill_npu_latency_pct) is not None]
    decode_npu_latency_pct = [pct for r in runs if (pct := r.decode_npu_latency_pct) is not None]
    total_npu_latency_pct = [pct for r in runs if (pct := r.total_npu_latency_pct) is not None]
    avg_power_w = [r.avg_power_w for r in runs if r.avg_power_w is not None]
    p99_power_w = [r.p99_power_w for r in runs if r.p99_power_w is not None]
    avg_utilization_pct = [r.avg_utilization_pct for r in runs if r.avg_utilization_pct is not None]
    p99_utilization_pct = [r.p99_utilization_pct for r in runs if r.p99_utilization_pct is not None]
    avg_temperature_c = [r.avg_temperature_c for r in runs if r.avg_temperature_c is not None]
    p99_temperature_c = [r.p99_temperature_c for r in runs if r.p99_temperature_c is not None]
    avg_memory_used_mb = [r.avg_memory_used_mb for r in runs if r.avg_memory_used_mb is not None]
    p99_memory_used_mb = [r.p99_memory_used_mb for r in runs if r.p99_memory_used_mb is not None]
    total_memory_mb = [r.total_memory_mb for r in runs if r.total_memory_mb is not None]
    avg_memory_used_pct = [r.avg_memory_used_pct for r in runs if r.avg_memory_used_pct is not None]
    p99_memory_used_pct = [r.p99_memory_used_pct for r in runs if r.p99_memory_used_pct is not None]
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
    prefill_avg_temperature_c = [r.prefill_avg_temperature_c for r in runs if r.prefill_avg_temperature_c is not None]
    prefill_p99_temperature_c = [r.prefill_p99_temperature_c for r in runs if r.prefill_p99_temperature_c is not None]
    decode_avg_temperature_c = [r.decode_avg_temperature_c for r in runs if r.decode_avg_temperature_c is not None]
    decode_p99_temperature_c = [r.decode_p99_temperature_c for r in runs if r.decode_p99_temperature_c is not None]
    prefill_avg_memory_used_mb = [
        r.prefill_avg_memory_used_mb for r in runs if r.prefill_avg_memory_used_mb is not None
    ]
    prefill_p99_memory_used_mb = [
        r.prefill_p99_memory_used_mb for r in runs if r.prefill_p99_memory_used_mb is not None
    ]
    decode_avg_memory_used_mb = [r.decode_avg_memory_used_mb for r in runs if r.decode_avg_memory_used_mb is not None]
    decode_p99_memory_used_mb = [r.decode_p99_memory_used_mb for r in runs if r.decode_p99_memory_used_mb is not None]
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
    acceptance_steps = [float(r.acceptance_steps) for r in runs if r.acceptance_steps is not None]
    acceptance_tokens_sum = [float(r.acceptance_tokens_sum) for r in runs if r.acceptance_tokens_sum is not None]
    acceptance_tokens_avg = [r.acceptance_tokens_avg for r in runs if r.acceptance_tokens_avg is not None]
    acceptance_ratio_pct = [
        (r.acceptance_ratio * 100.0) for r in runs if r.acceptance_ratio is not None
    ]

    print(f"warmup: {args.warmup}")
    print(f"runs: {args.repeat}")
    print(f"batch size: {batch_size}")
    print(f"prefill tokens: {runs[0].num_prefill} | decode tokens: {runs[0].num_decode}")
    _print_summary_header()
    _print_summary("prefill_tps", prefill_tps, "tok/s")
    _print_summary("decode_tps", decode_tps, "tok/s")
    _print_summary("ttft", ttft_ms, "ms")
    _print_summary("decode_duration", decode_ms, "ms")
    _print_summary("total", total_ms, "ms")
    _print_summary("prefill_npu_latency", prefill_npu_latency_pct, "%")
    _print_summary("decode_npu_latency", decode_npu_latency_pct, "%")
    _print_summary("total_npu_latency", total_npu_latency_pct, "%")
    _print_summary("accept_steps", acceptance_steps, "count")
    _print_summary("accept_tok_sum", acceptance_tokens_sum, "tok")
    _print_summary("accept_tok_avg", acceptance_tokens_avg, "tok")
    _print_summary("accept_ratio", acceptance_ratio_pct, "%")
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
        _print_summary("avg_temperature", avg_temperature_c, "C")
        _print_summary("p99_temperature", p99_temperature_c, "C")
        _print_summary("prefill_avg_temp", prefill_avg_temperature_c, "C")
        _print_summary("prefill_p99_temp", prefill_p99_temperature_c, "C")
        _print_summary("decode_avg_temp", decode_avg_temperature_c, "C")
        _print_summary("decode_p99_temp", decode_p99_temperature_c, "C")
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
            "batch_size": batch_size,
            "input": {
                "mode": str(getattr(args, "input_mode", "random")),
                "prompt_sha256": selected_prompt_sha256,
            },
            "runs": [asdict(r) for r in runs],
            "summary": {
                "prefill_tps": _summary(prefill_tps),
                "decode_tps": _summary(decode_tps),
                "ttft_ms": _summary(ttft_ms),
                "decode_duration_ms": _summary(decode_ms),
                "total_ms": _summary(total_ms),
                "prefill_npu_latency_pct": _summary(prefill_npu_latency_pct),
                "decode_npu_latency_pct": _summary(decode_npu_latency_pct),
                "total_npu_latency_pct": _summary(total_npu_latency_pct),
                "acceptance_steps": _summary(acceptance_steps),
                "acceptance_tokens_sum": _summary(acceptance_tokens_sum),
                "acceptance_tokens_avg": _summary(acceptance_tokens_avg),
                "acceptance_ratio_pct": _summary(acceptance_ratio_pct),
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
                "avg_temperature_c": _summary(avg_temperature_c),
                "p99_temperature_c": _summary(p99_temperature_c),
                "prefill_avg_temperature_c": _summary(prefill_avg_temperature_c),
                "prefill_p99_temperature_c": _summary(prefill_p99_temperature_c),
                "decode_avg_temperature_c": _summary(decode_avg_temperature_c),
                "decode_p99_temperature_c": _summary(decode_p99_temperature_c),
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
            "device_time_series_runs": run_phase_device_time_series,
        }
        _write_json(args.json, payload)
        print(f"wrote: {args.json}")

    return 0


def _run_vlm_measure(args: argparse.Namespace) -> int:
    """Run a single VLM TPS measurement with separate vision and LLM metrics."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    _normalize_runtime_defaults(args)
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
        eagle3_options=_extract_eagle3_pipeline_kwargs(args),
        target_cores=args.target_cores,
        target_clusters=args.target_clusters,
    )
    batch_size = _resolve_cli_batch_size(args, pipeline)

    from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
        SingleMeasurement,
        VLMSingleMeasurement,
        VLMTPSMeasurer,
    )

    def _single_llm_measurement(result: Any) -> SingleMeasurement:
        """Convert a fixed-point VLM LLM sweep result into a single measurement."""
        if not result.prefill_sweep.x_values:
            raise SystemExit(
                f"Requested VLM prefill length {args.prefill} is shorter than the model's multimodal prefix length."
            )
        if not result.decode_sweep.x_values:
            raise SystemExit(f"Requested VLM decode cache length {args.prefill} could not be measured.")

        prefill_idx = len(result.prefill_sweep.x_values) - 1
        decode_idx = len(result.decode_sweep.x_values) - 1
        prefill_latency = float(result.prefill_sweep.time_values[prefill_idx])
        decode_duration = float(result.decode_sweep.time_values[decode_idx])
        avg_total_prefill = result.prefill_sweep.avg_total_token_latency_values[prefill_idx]
        avg_npu_prefill = result.prefill_sweep.avg_npu_token_latency_values[prefill_idx]
        avg_total_decode = result.decode_sweep.avg_total_token_latency_values[decode_idx]
        avg_npu_decode = result.decode_sweep.avg_npu_token_latency_values[decode_idx]
        npu_prefill_time = None if avg_npu_prefill is None else float(avg_npu_prefill) * int(args.prefill) * batch_size
        npu_decode_time = None if avg_npu_decode is None else float(avg_npu_decode) * int(args.decode) * batch_size
        total_npu_time = (
            npu_prefill_time + npu_decode_time if npu_prefill_time is not None and npu_decode_time is not None else None
        )
        return SingleMeasurement(
            num_prefill=int(result.prefill_sweep.x_values[prefill_idx]),
            num_decode=int(args.decode),
            prefill_latency=prefill_latency,
            prefill_tps=float(result.prefill_sweep.tps_values[prefill_idx]),
            decode_duration=decode_duration,
            decode_tps=float(result.decode_sweep.tps_values[decode_idx]),
            total_time=prefill_latency + decode_duration,
            avg_total_prefill_token_latency=float(avg_total_prefill or 0.0),
            avg_npu_prefill_token_latency=avg_npu_prefill,
            avg_total_decode_token_latency=float(avg_total_decode or 0.0),
            avg_npu_decode_token_latency=avg_npu_decode,
            prefill_npu_latency_pct=npu_latency_pct(avg_total_prefill, avg_npu_prefill),
            decode_npu_latency_pct=npu_latency_pct(avg_total_decode, avg_npu_decode),
            total_npu_latency_pct=npu_latency_pct(prefill_latency + decode_duration, total_npu_time),
            npu_prefill_time=npu_prefill_time,
            npu_decode_time=npu_decode_time,
            decode_prefill_mode=(result.decode_prefill_modes[decode_idx] if result.decode_prefill_modes else "real"),
        )

    def _measure_fixed_vlm_run(*, show_progress: bool) -> VLMSingleMeasurement:
        """Measure VLM vision plus LLM with the requested fixed prefill length."""
        vision_latency, vision_fps = measurer.measure_vision(
            image_resolution=args.image_resolution,
            repeat=1,
            prompt=args.prompt,
            batch_size=batch_size,
            show_progress=show_progress,
        )[0]
        llm_result = measurer.measure_llm_full(
            image_resolution=args.image_resolution,
            prompt=args.prompt,
            prefill_range=(args.prefill, args.prefill, args.prefill),
            cache_lengths=[args.prefill],
            decode_window=args.decode,
            prefill_chunk_size=args.prefill_chunk_size,
            show_progress=show_progress,
            batch_size=batch_size,
        )
        return VLMSingleMeasurement(
            image_resolution=args.image_resolution,
            vision_encode_latency=vision_latency,
            vision_fps=vision_fps,
            llm=_single_llm_measurement(llm_result),
        )

    measurer = VLMTPSMeasurer(pipeline)
    tracker = _build_device_tracker(args, pipeline)
    _print_device_status(args, tracker)

    for i in tqdm(range(args.warmup), desc="warmup runs", leave=False):
        _measure_fixed_vlm_run(show_progress=False)

    runs = []
    device_metrics = []
    device_time_series_runs: list[dict[str, list[dict[str, float]]]] = []
    for _ in tqdm(range(args.repeat), desc="measure runs", leave=False):
        if tracker is not None:
            tracker.start()
        try:
            run = _measure_fixed_vlm_run(show_progress=False)
        finally:
            _stop_tracker_safe(tracker)
        runs.append(run)
        if tracker is not None:
            metric = _extract_device_metric(tracker)
            avg_power = metric.get("avg_power_w")
            if avg_power is not None:
                metric["total_energy_j"] = avg_power * ((run.vision_encode_latency * batch_size) + run.llm.total_time)
            device_metrics.append(metric)
            device_time_series_runs.append(_extract_device_time_series(tracker))

    vision_ms = [r.vision_encode_latency * 1000.0 for r in runs]
    vision_fps = [r.vision_fps for r in runs]
    prefill_tps = [r.llm.prefill_tps for r in runs]
    decode_tps = [r.llm.decode_tps for r in runs]
    ttft_ms = [r.llm.prefill_latency * 1000.0 for r in runs]
    decode_ms = [r.llm.decode_duration * 1000.0 for r in runs]
    total_ms = [((r.vision_encode_latency * batch_size) + r.llm.total_time) * 1000.0 for r in runs]
    prefill_npu_latency_pct = [pct for r in runs if (pct := r.llm.prefill_npu_latency_pct) is not None]
    decode_npu_latency_pct = [pct for r in runs if (pct := r.llm.decode_npu_latency_pct) is not None]
    total_npu_latency_pct = [pct for r in runs if (pct := r.llm.total_npu_latency_pct) is not None]

    avg_power_w = [m["avg_power_w"] for m in device_metrics if m.get("avg_power_w") is not None]
    p99_power_w = [m["p99_power_w"] for m in device_metrics if m.get("p99_power_w") is not None]
    avg_utilization_pct = [m["avg_utilization_pct"] for m in device_metrics if m.get("avg_utilization_pct") is not None]
    p99_utilization_pct = [m["p99_utilization_pct"] for m in device_metrics if m.get("p99_utilization_pct") is not None]
    avg_temperature_c = [m["avg_temperature_c"] for m in device_metrics if m.get("avg_temperature_c") is not None]
    p99_temperature_c = [m["p99_temperature_c"] for m in device_metrics if m.get("p99_temperature_c") is not None]
    avg_memory_used_mb = [m["avg_memory_used_mb"] for m in device_metrics if m.get("avg_memory_used_mb") is not None]
    p99_memory_used_mb = [m["p99_memory_used_mb"] for m in device_metrics if m.get("p99_memory_used_mb") is not None]
    total_memory_mb = [m["total_memory_mb"] for m in device_metrics if m.get("total_memory_mb") is not None]
    avg_memory_used_pct = [m["avg_memory_used_pct"] for m in device_metrics if m.get("avg_memory_used_pct") is not None]
    p99_memory_used_pct = [m["p99_memory_used_pct"] for m in device_metrics if m.get("p99_memory_used_pct") is not None]
    total_energy_j = [m["total_energy_j"] for m in device_metrics if m.get("total_energy_j") is not None]

    print(f"warmup: {args.warmup}")
    print(f"runs: {args.repeat}")
    print(f"batch size: {batch_size}")
    print(
        f"image resolution: {args.image_resolution} | "
        f"llm prefill tokens: {runs[0].llm.num_prefill} | decode tokens: {runs[0].llm.num_decode}"
    )
    _print_summary_header()
    _print_summary("vision_encode", vision_ms, "ms")
    _print_summary("vision_fps", vision_fps, "fps")
    _print_summary("llm_prefill_tps", prefill_tps, "tok/s")
    _print_summary("llm_decode_tps", decode_tps, "tok/s")
    _print_summary("llm_ttft", ttft_ms, "ms")
    _print_summary("llm_decode_duration", decode_ms, "ms")
    _print_summary("total", total_ms, "ms")
    _print_summary("llm_prefill_npu_lat", prefill_npu_latency_pct, "%")
    _print_summary("llm_decode_npu_lat", decode_npu_latency_pct, "%")
    _print_summary("llm_total_npu_lat", total_npu_latency_pct, "%")
    if args.device_metrics:
        _print_summary("avg_power", avg_power_w, "W")
        _print_summary("p99_power", p99_power_w, "W")
        _print_summary("avg_utilization", avg_utilization_pct, "%")
        _print_summary("p99_utilization", p99_utilization_pct, "%")
        _print_summary("avg_temperature", avg_temperature_c, "C")
        _print_summary("p99_temperature", p99_temperature_c, "C")
        _print_summary("avg_memory_used", avg_memory_used_mb, "MB")
        _print_summary("p99_memory_used", p99_memory_used_mb, "MB")
        _print_summary("total_memory", total_memory_mb, "MB")
        _print_summary("avg_memory_used_pct", avg_memory_used_pct, "%")
        _print_summary("p99_memory_used_pct", p99_memory_used_pct, "%")
        _print_summary("total_energy", total_energy_j, "J")
    _print_summary_footer()

    if args.json:
        payload = {
            "task": args.task,
            "model": args.model,
            "prompt": args.prompt,
            "image_resolution": args.image_resolution,
            "repeat": args.repeat,
            "batch_size": batch_size,
            "runs": [asdict(r) for r in runs],
            "summary": {
                "vision_encode_ms": _summary(vision_ms),
                "vision_fps": _summary(vision_fps),
                "llm_prefill_tps": _summary(prefill_tps),
                "llm_decode_tps": _summary(decode_tps),
                "llm_ttft_ms": _summary(ttft_ms),
                "llm_decode_duration_ms": _summary(decode_ms),
                "total_ms": _summary(total_ms),
                "llm_prefill_npu_latency_pct": _summary(prefill_npu_latency_pct),
                "llm_decode_npu_latency_pct": _summary(decode_npu_latency_pct),
                "llm_total_npu_latency_pct": _summary(total_npu_latency_pct),
                "avg_power_w": _summary(avg_power_w),
                "p99_power_w": _summary(p99_power_w),
                "avg_utilization_pct": _summary(avg_utilization_pct),
                "p99_utilization_pct": _summary(p99_utilization_pct),
                "avg_temperature_c": _summary(avg_temperature_c),
                "p99_temperature_c": _summary(p99_temperature_c),
                "avg_memory_used_mb": _summary(avg_memory_used_mb),
                "p99_memory_used_mb": _summary(p99_memory_used_mb),
                "total_memory_mb": _summary(total_memory_mb),
                "avg_memory_used_pct": _summary(avg_memory_used_pct),
                "p99_memory_used_pct": _summary(p99_memory_used_pct),
                "total_energy_j": _summary(total_energy_j),
            },
            "device_runs": device_metrics,
            "device_time_series_runs": device_time_series_runs,
        }
        _write_json(args.json, payload)
        print(f"wrote: {args.json}")

    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Dispatch a TPS sweep to the text or VLM measurement path."""
    if _is_vlm_task(args.task):
        return _run_vlm_sweep(args)
    return _run_text_sweep(args)


def _run_text_sweep(args: argparse.Namespace) -> int:
    """Run a text-generation TPS prefill/decode sweep."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    _normalize_runtime_defaults(args)
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
        eagle3_options=_extract_eagle3_pipeline_kwargs(args),
        target_cores=args.target_cores,
        target_clusters=args.target_clusters,
    )
    batch_size = _resolve_cli_batch_size(args, pipeline)

    from mblt_model_zoo.hf_transformers.utils.benchmark_utils import TPSMeasurer

    measurer = TPSMeasurer(pipeline)
    tracker_prefill, tracker_decode = _build_phase_trackers(args, pipeline)
    _print_device_status(args, tracker_prefill)
    for i in tqdm(range(args.warmup), desc="warmup runs", leave=False):
        measurer.measure(
            num_prefill=_SWEEP_WARMUP_PREFILL,
            num_decode=_SWEEP_WARMUP_DECODE,
            prefill_chunk_size=args.prefill_chunk_size,
            trace_path=None,
            show_progress=True,
            progress_desc=f"warmup generate {i + 1}/{args.warmup}",
            batch_size=batch_size,
        )
    runs = []
    run_avg_power: list[float] = []
    run_p99_power: list[float] = []
    run_avg_utilization: list[float] = []
    run_p99_utilization: list[float] = []
    run_avg_temperature: list[float] = []
    run_p99_temperature: list[float] = []
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
    run_prefill_avg_temp: list[float] = []
    run_prefill_p99_temp: list[float] = []
    run_decode_avg_temp: list[float] = []
    run_decode_p99_temp: list[float] = []
    run_prefill_avg_mem_used_mb: list[float] = []
    run_prefill_p99_mem_used_mb: list[float] = []
    run_decode_avg_mem_used_mb: list[float] = []
    run_decode_p99_mem_used_mb: list[float] = []
    run_prefill_avg_mem_used_pct: list[float] = []
    run_prefill_p99_mem_used_pct: list[float] = []
    run_decode_avg_mem_used_pct: list[float] = []
    run_decode_p99_mem_used_pct: list[float] = []
    run_phase_device: list[dict[str, dict[str, Optional[float]]]] = []
    run_phase_device_time_series: list[dict[str, dict[str, list[dict[str, float]]]]] = []
    for i in tqdm(range(args.repeat), desc="sweep runs", leave=False):
        prefill_metric: dict[str, Optional[float]] = {}
        decode_metric: dict[str, Optional[float]] = {}
        try:
            runs.append(
                measurer.measure_full(
                    prefill_range=args.prefill_range,
                    cache_lengths=args.cache_lengths,
                    decode_window=args.decode_window,
                    prefill_chunk_size=args.prefill_chunk_size,
                    trace_path=args.trace if i == 0 else None,
                    show_progress=True,
                    progress_prefix=f"run {i + 1}/{args.repeat}",
                    on_prefill_start=(lambda: tracker_prefill.start()) if tracker_prefill is not None else None,
                    on_prefill_end=(lambda: tracker_prefill.stop()) if tracker_prefill is not None else None,
                    on_decode_start=(lambda: tracker_decode.start()) if tracker_decode is not None else None,
                    on_decode_end=(lambda: tracker_decode.stop()) if tracker_decode is not None else None,
                    batch_size=batch_size,
                )
            )
        finally:
            _stop_tracker_safe(tracker_prefill)
            _stop_tracker_safe(tracker_decode)
        if tracker_prefill is not None and tracker_decode is not None:
            prefill_metric = _extract_device_metric(tracker_prefill)
            decode_metric = _extract_device_metric(tracker_decode)
            run_phase_device.append({"prefill": prefill_metric, "decode": decode_metric})
            run_phase_device_time_series.append(
                {
                    "prefill": _extract_device_time_series(tracker_prefill),
                    "decode": _extract_device_time_series(tracker_decode),
                }
            )
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
            avg_temp = _weighted_two(
                prefill_metric.get("avg_temperature_c"),
                prefill_dur,
                decode_metric.get("avg_temperature_c"),
                decode_dur,
            )
            if avg_temp is not None:
                run_avg_temperature.append(float(avg_temp))
            p99_temp = max(
                [
                    v
                    for v in (
                        prefill_metric.get("p99_temperature_c"),
                        decode_metric.get("p99_temperature_c"),
                    )
                    if v is not None
                ],
                default=None,
            )
            if p99_temp is not None:
                run_p99_temperature.append(float(p99_temp))
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
                [
                    v
                    for v in (prefill_metric.get("total_memory_mb"), decode_metric.get("total_memory_mb"))
                    if v is not None
                ],
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
                (run_prefill_avg_temp, "avg_temperature_c"),
                (run_prefill_p99_temp, "p99_temperature_c"),
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
                (run_decode_avg_temp, "avg_temperature_c"),
                (run_decode_p99_temp, "p99_temperature_c"),
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
    prefill_npu_latency_pct_last = [
        pct
        for r in runs
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
    decode_npu_latency_pct_last = [
        pct
        for r in runs
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
    print(f"batch size: {batch_size}")
    _print_summary_header()
    if prefill_last:
        _print_summary("prefill_tps(last_point)", prefill_last, "tok/s")
    if decode_last:
        _print_summary("decode_tps(last_point)", decode_last, "tok/s")
    _print_summary("prefill_npu_lat(last)", prefill_npu_latency_pct_last, "%")
    _print_summary("decode_npu_lat(last)", decode_npu_latency_pct_last, "%")
    if args.device_metrics:
        _print_summary("avg_power", run_avg_power, "W")
        _print_summary("p99_power", run_p99_power, "W")
        _print_summary("prefill_avg_power", run_prefill_avg_power, "W")
        _print_summary("prefill_p99_power", run_prefill_p99_power, "W")
        _print_summary("decode_avg_power", run_decode_avg_power, "W")
        _print_summary("decode_p99_power", run_decode_p99_power, "W")
        _print_summary("avg_utilization", run_avg_utilization, "%")
        _print_summary("p99_utilization", run_p99_utilization, "%")
        _print_summary("avg_temperature", run_avg_temperature, "C")
        _print_summary("p99_temperature", run_p99_temperature, "C")
        _print_summary("prefill_avg_util", run_prefill_avg_util, "%")
        _print_summary("prefill_p99_util", run_prefill_p99_util, "%")
        _print_summary("decode_avg_util", run_decode_avg_util, "%")
        _print_summary("decode_p99_util", run_decode_p99_util, "%")
        _print_summary("prefill_avg_temp", run_prefill_avg_temp, "C")
        _print_summary("prefill_p99_temp", run_prefill_p99_temp, "C")
        _print_summary("decode_avg_temp", run_decode_avg_temp, "C")
        _print_summary("decode_p99_temp", run_decode_p99_temp, "C")
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
            "batch_size": batch_size,
            "aggregate": asdict(result),
            "runs": [asdict(r) for r in runs],
            "summary": {
                "prefill_tps_last": _summary(prefill_last),
                "decode_tps_last": _summary(decode_last),
                "prefill_npu_latency_pct_last": _summary(prefill_npu_latency_pct_last),
                "decode_npu_latency_pct_last": _summary(decode_npu_latency_pct_last),
                "avg_power_w": _summary(run_avg_power),
                "p99_power_w": _summary(run_p99_power),
                "avg_utilization_pct": _summary(run_avg_utilization),
                "p99_utilization_pct": _summary(run_p99_utilization),
                "avg_temperature_c": _summary(run_avg_temperature),
                "p99_temperature_c": _summary(run_p99_temperature),
                "prefill_avg_power_w": _summary(run_prefill_avg_power),
                "prefill_p99_power_w": _summary(run_prefill_p99_power),
                "decode_avg_power_w": _summary(run_decode_avg_power),
                "decode_p99_power_w": _summary(run_decode_p99_power),
                "prefill_avg_utilization_pct": _summary(run_prefill_avg_util),
                "prefill_p99_utilization_pct": _summary(run_prefill_p99_util),
                "decode_avg_utilization_pct": _summary(run_decode_avg_util),
                "decode_p99_utilization_pct": _summary(run_decode_p99_util),
                "prefill_avg_temperature_c": _summary(run_prefill_avg_temp),
                "prefill_p99_temperature_c": _summary(run_prefill_p99_temp),
                "decode_avg_temperature_c": _summary(run_decode_avg_temp),
                "decode_p99_temperature_c": _summary(run_decode_p99_temp),
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
            "device_time_series_runs": run_phase_device_time_series,
        }
        _write_json(args.json, payload)
        print(f"wrote: {args.json}")

    if args.csv:
        rows = list(_iter_rows_for_csv(result))
        for row in rows:
            row["batch_size"] = batch_size
        _write_csv(args.csv, rows)
        print(f"wrote: {args.csv}")

    if args.plot:
        measurer.plot_and_save(result, save_path=args.plot)

    return 0


def _plot_vlm_sweep(
    *,
    resolution_payloads: Sequence[dict[str, Any]],
    llm_result: Any,
    save_path: str,
) -> None:
    """Write a VLM sweep summary plot.

    Args:
        resolution_payloads: Vision sweep payloads produced by ``_run_vlm_sweep``.
        llm_result: Aggregated LLM TPS sweep result.
        save_path: Destination PNG path.
    """
    import matplotlib.pyplot as plt

    output_dir = os.path.dirname(os.path.abspath(save_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    resolutions = [int(payload["image_resolution"]) for payload in resolution_payloads]
    vision_encode_ms = [payload["summary"]["vision_encode_ms"]["mean"] for payload in resolution_payloads]
    vision_fps = [payload["summary"]["vision_fps"]["mean"] for payload in resolution_payloads]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("VLM TPS Sweep", fontsize=16)

    axs[0, 0].plot(resolutions, vision_encode_ms, "o-", color="tab:red")
    axs[0, 0].set_title("Vision Encode Latency")
    axs[0, 0].set_xlabel("Image resolution")
    axs[0, 0].set_ylabel("Latency (ms/image)")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(resolutions, vision_fps, "o-", color="tab:blue")
    axs[0, 1].set_title("Vision Throughput")
    axs[0, 1].set_xlabel("Image resolution")
    axs[0, 1].set_ylabel("FPS")
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(llm_result.prefill_sweep.x_values, llm_result.prefill_sweep.tps_values, "o-", color="tab:green")
    axs[1, 0].set_title("LLM Prefill TPS")
    axs[1, 0].set_xlabel("Prefill tokens")
    axs[1, 0].set_ylabel("Tokens/s")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(llm_result.decode_sweep.x_values, llm_result.decode_sweep.tps_values, "o-", color="tab:purple")
    axs[1, 1].set_title("LLM Decode TPS")
    axs[1, 1].set_xlabel("Cache length")
    axs[1, 1].set_ylabel("Tokens/s")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _run_vlm_sweep(args: argparse.Namespace) -> int:
    """Run a VLM TPS sweep including vision encoder and LLM phases."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    _normalize_runtime_defaults(args)
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
        eagle3_options=_extract_eagle3_pipeline_kwargs(args),
        target_cores=args.target_cores,
        target_clusters=args.target_clusters,
    )
    batch_size = _resolve_cli_batch_size(args, pipeline)

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
                batch_size=batch_size,
            )
        vision_runs = []
        vision_power_avg = []
        vision_power_p99 = []
        vision_util_avg = []
        vision_util_p99 = []
        vision_temp_avg = []
        vision_temp_p99 = []
        vision_mem_used_avg_mb = []
        vision_mem_used_p99_mb = []
        vision_mem_total_mb = []
        vision_mem_used_pct_avg = []
        vision_mem_used_pct_p99 = []
        vision_energy_j = []
        vision_img_per_j = []
        vision_j_per_img = []
        vision_device_time_series_runs: list[dict[str, list[dict[str, float]]]] = []
        for _ in tqdm(range(args.repeat), desc=f"vision@{resolution}", leave=False):
            if tracker is not None:
                tracker.start()
            try:
                single = measurer.measure_vision(
                    image_resolution=resolution,
                    repeat=1,
                    prompt=args.prompt,
                    show_progress=False,
                    batch_size=batch_size,
                )[0]
            finally:
                _stop_tracker_safe(tracker)
            vision_runs.append(single)
            if tracker is not None:
                metric = _extract_device_metric(tracker)
                vision_device_time_series_runs.append(_extract_device_time_series(tracker))
                avg_power = metric.get("avg_power_w")
                p99_power = metric.get("p99_power_w")
                avg_utilization = metric.get("avg_utilization_pct")
                p99_utilization = metric.get("p99_utilization_pct")
                avg_temperature = metric.get("avg_temperature_c")
                p99_temperature = metric.get("p99_temperature_c")
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
                if avg_temperature is not None:
                    vision_temp_avg.append(float(avg_temperature))
                if p99_temperature is not None:
                    vision_temp_p99.append(float(p99_temperature))
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

        print(f"\nresolution={resolution} warmup={args.warmup} runs={args.repeat} batch_size={batch_size}")
        _print_summary_header()
        _print_summary("vision_encode", vision_ms, "ms")
        _print_summary("vision_fps", vision_fps, "fps")
        if args.device_metrics:
            _print_summary("vision_avg_power", vision_power_avg, "W")
            _print_summary("vision_p99_power", vision_power_p99, "W")
            _print_summary("vision_avg_util", vision_util_avg, "%")
            _print_summary("vision_p99_util", vision_util_p99, "%")
            _print_summary("vision_avg_temp", vision_temp_avg, "C")
            _print_summary("vision_p99_temp", vision_temp_p99, "C")
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
                    "batch_size": batch_size,
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
                    "llm_prefill_npu_latency_pct": None,
                    "llm_decode_npu_latency_pct": None,
                    "avg_power_w": vision_power_avg[idx - 1] if idx - 1 < len(vision_power_avg) else None,
                    "p99_power_w": vision_power_p99[idx - 1] if idx - 1 < len(vision_power_p99) else None,
                    "avg_utilization_pct": vision_util_avg[idx - 1] if idx - 1 < len(vision_util_avg) else None,
                    "p99_utilization_pct": vision_util_p99[idx - 1] if idx - 1 < len(vision_util_p99) else None,
                    "avg_temperature_c": vision_temp_avg[idx - 1] if idx - 1 < len(vision_temp_avg) else None,
                    "p99_temperature_c": vision_temp_p99[idx - 1] if idx - 1 < len(vision_temp_p99) else None,
                    "avg_memory_used_mb": vision_mem_used_avg_mb[idx - 1]
                    if idx - 1 < len(vision_mem_used_avg_mb)
                    else None,
                    "p99_memory_used_mb": vision_mem_used_p99_mb[idx - 1]
                    if idx - 1 < len(vision_mem_used_p99_mb)
                    else None,
                    "total_memory_mb": vision_mem_total_mb[idx - 1] if idx - 1 < len(vision_mem_total_mb) else None,
                    "avg_memory_used_pct": vision_mem_used_pct_avg[idx - 1]
                    if idx - 1 < len(vision_mem_used_pct_avg)
                    else None,
                    "p99_memory_used_pct": vision_mem_used_pct_p99[idx - 1]
                    if idx - 1 < len(vision_mem_used_pct_p99)
                    else None,
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
                "batch_size": batch_size,
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
                    "avg_temperature_c": _summary(vision_temp_avg),
                    "p99_temperature_c": _summary(vision_temp_p99),
                    "avg_memory_used_mb": _summary(vision_mem_used_avg_mb),
                    "p99_memory_used_mb": _summary(vision_mem_used_p99_mb),
                    "total_memory_mb": _summary(vision_mem_total_mb),
                    "avg_memory_used_pct": _summary(vision_mem_used_pct_avg),
                    "p99_memory_used_pct": _summary(vision_mem_used_pct_p99),
                    "energy_j": _summary(vision_energy_j),
                    "vision_img_per_j": _summary(vision_img_per_j),
                    "vision_j_per_img": _summary(vision_j_per_img),
                },
                "device_time_series_runs": vision_device_time_series_runs,
            }
        )

    llm_resolution = args.llm_resolution if args.llm_resolution is not None else args.image_resolutions[0]
    for _ in range(args.warmup):
        measurer.measure_llm_full(
            image_resolution=llm_resolution,
            prompt=args.prompt,
            prefill_range=args.prefill_range,
            cache_lengths=args.cache_lengths,
            decode_window=args.decode_window,
            show_progress=False,
            batch_size=batch_size,
        )
    llm_runs = []
    llm_device_time_series_runs: list[dict[str, list[dict[str, float]]]] = []
    for _ in tqdm(range(args.repeat), desc=f"llm@{llm_resolution}", leave=False):
        if tracker is not None:
            tracker.start()
        try:
            run = measurer.measure_llm_full(
                image_resolution=llm_resolution,
                prompt=args.prompt,
                prefill_range=args.prefill_range,
                cache_lengths=args.cache_lengths,
                decode_window=args.decode_window,
                show_progress=False,
                batch_size=batch_size,
            )
        finally:
            _stop_tracker_safe(tracker)
        if tracker is not None and (run.prefill_sweep.x_values or run.decode_sweep.x_values):
            metric = _extract_device_metric(tracker)
            llm_device_time_series_runs.append(_extract_device_time_series(tracker))
            _enrich_single_run_device(
                run=run,
                prefill_metric=metric,
                decode_metric=metric,
                batch_size=batch_size,
            )
        llm_runs.append(run)

    llm_result = _aggregate_sweep_results(llm_runs)
    llm_prefill_tps = [r.prefill_sweep.tps_values[-1] for r in llm_runs if r.prefill_sweep.tps_values]
    llm_decode_tps = [r.decode_sweep.tps_values[-1] for r in llm_runs if r.decode_sweep.tps_values]
    llm_ttft_ms = [r.prefill_sweep.time_values[-1] * 1000.0 for r in llm_runs if r.prefill_sweep.time_values]
    llm_decode_ms = [r.decode_sweep.time_values[-1] * 1000.0 for r in llm_runs if r.decode_sweep.time_values]
    llm_prefill_npu_latency_pct = [
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
    llm_decode_npu_latency_pct = [
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
    llm_avg_power_w = [r.avg_power_w for r in llm_runs if getattr(r, "avg_power_w", None) is not None]
    llm_p99_power_w = [r.p99_power_w for r in llm_runs if getattr(r, "p99_power_w", None) is not None]
    llm_avg_utilization_pct = [
        r.avg_utilization_pct for r in llm_runs if getattr(r, "avg_utilization_pct", None) is not None
    ]
    llm_p99_utilization_pct = [
        r.p99_utilization_pct for r in llm_runs if getattr(r, "p99_utilization_pct", None) is not None
    ]
    llm_avg_temperature_c = [r.avg_temperature_c for r in llm_runs if getattr(r, "avg_temperature_c", None) is not None]
    llm_p99_temperature_c = [r.p99_temperature_c for r in llm_runs if getattr(r, "p99_temperature_c", None) is not None]
    llm_avg_memory_used_mb = [
        r.avg_memory_used_mb for r in llm_runs if getattr(r, "avg_memory_used_mb", None) is not None
    ]
    llm_p99_memory_used_mb = [
        r.p99_memory_used_mb for r in llm_runs if getattr(r, "p99_memory_used_mb", None) is not None
    ]
    llm_total_memory_mb = [r.total_memory_mb for r in llm_runs if getattr(r, "total_memory_mb", None) is not None]
    llm_avg_memory_used_pct = [
        r.avg_memory_used_pct for r in llm_runs if getattr(r, "avg_memory_used_pct", None) is not None
    ]
    llm_p99_memory_used_pct = [
        r.p99_memory_used_pct for r in llm_runs if getattr(r, "p99_memory_used_pct", None) is not None
    ]
    llm_total_energy_j = [r.total_energy_j for r in llm_runs if getattr(r, "total_energy_j", None) is not None]

    print(
        f"\nllm_reference_resolution={llm_resolution} warmup={args.warmup} runs={args.repeat} batch_size={batch_size}"
    )
    _print_summary_header()
    _print_summary("llm_prefill_tps(last)", llm_prefill_tps, "tok/s")
    _print_summary("llm_decode_tps(last)", llm_decode_tps, "tok/s")
    _print_summary("llm_ttft(last)", llm_ttft_ms, "ms")
    _print_summary("llm_decode_duration(last)", llm_decode_ms, "ms")
    _print_summary("llm_prefill_npu_lat", llm_prefill_npu_latency_pct, "%")
    _print_summary("llm_decode_npu_lat", llm_decode_npu_latency_pct, "%")
    if args.device_metrics:
        _print_summary("llm_avg_power", llm_avg_power_w, "W")
        _print_summary("llm_p99_power", llm_p99_power_w, "W")
        _print_summary("llm_avg_utilization", llm_avg_utilization_pct, "%")
        _print_summary("llm_p99_utilization", llm_p99_utilization_pct, "%")
        _print_summary("llm_avg_temperature", llm_avg_temperature_c, "C")
        _print_summary("llm_p99_temperature", llm_p99_temperature_c, "C")
        _print_summary("llm_avg_mem_used", llm_avg_memory_used_mb, "MB")
        _print_summary("llm_p99_mem_used", llm_p99_memory_used_mb, "MB")
        _print_summary("llm_total_mem", llm_total_memory_mb, "MB")
        _print_summary("llm_avg_mem_used_pct", llm_avg_memory_used_pct, "%")
        _print_summary("llm_p99_mem_used_pct", llm_p99_memory_used_pct, "%")
        _print_summary("llm_total_energy", llm_total_energy_j, "J")
        if not llm_avg_power_w:
            print("[device] warning: no llm device samples were collected")
    _print_summary_footer()

    for idx, run in enumerate(llm_runs, start=1):
        prefill_tokens = run.prefill_sweep.x_values[-1] if run.prefill_sweep.x_values else None
        decode_tokens = run.decode_sweep.x_values[-1] if run.decode_sweep.x_values else None
        prefill_tps = run.prefill_sweep.tps_values[-1] if run.prefill_sweep.tps_values else None
        decode_tps = run.decode_sweep.tps_values[-1] if run.decode_sweep.tps_values else None
        ttft_ms = (run.prefill_sweep.time_values[-1] * 1000.0) if run.prefill_sweep.time_values else None
        decode_ms = (run.decode_sweep.time_values[-1] * 1000.0) if run.decode_sweep.time_values else None
        prefill_npu_pct = None
        if run.prefill_sweep.avg_total_token_latency_values and run.prefill_sweep.avg_npu_token_latency_values:
            prefill_npu_pct = npu_latency_pct(
                run.prefill_sweep.avg_total_token_latency_values[-1],
                run.prefill_sweep.avg_npu_token_latency_values[-1],
            )
        decode_npu_pct = None
        if run.decode_sweep.avg_total_token_latency_values and run.decode_sweep.avg_npu_token_latency_values:
            decode_npu_pct = npu_latency_pct(
                run.decode_sweep.avg_total_token_latency_values[-1],
                run.decode_sweep.avg_npu_token_latency_values[-1],
            )
        csv_rows.append(
            {
                "type": "llm",
                "batch_size": batch_size,
                "image_resolution": llm_resolution,
                "repeat_index": idx,
                "vision_encode_ms": None,
                "vision_fps": None,
                "llm_prefill_tokens": prefill_tokens,
                "llm_decode_tokens": decode_tokens,
                "llm_prefill_tps": prefill_tps,
                "llm_decode_tps": decode_tps,
                "llm_ttft_ms": ttft_ms,
                "llm_decode_ms": decode_ms,
                "llm_total_ms": None,
                "llm_prefill_npu_latency_pct": prefill_npu_pct,
                "llm_decode_npu_latency_pct": decode_npu_pct,
                "avg_power_w": getattr(run, "avg_power_w", None),
                "p99_power_w": getattr(run, "p99_power_w", None),
                "avg_utilization_pct": getattr(run, "avg_utilization_pct", None),
                "p99_utilization_pct": getattr(run, "p99_utilization_pct", None),
                "avg_temperature_c": getattr(run, "avg_temperature_c", None),
                "p99_temperature_c": getattr(run, "p99_temperature_c", None),
                "avg_memory_used_mb": getattr(run, "avg_memory_used_mb", None),
                "p99_memory_used_mb": getattr(run, "p99_memory_used_mb", None),
                "total_memory_mb": getattr(run, "total_memory_mb", None),
                "avg_memory_used_pct": getattr(run, "avg_memory_used_pct", None),
                "p99_memory_used_pct": getattr(run, "p99_memory_used_pct", None),
                "total_energy_j": getattr(run, "total_energy_j", None),
                "prefill_tok_per_j": None,
                "decode_tok_per_j": None,
                "prefill_j_per_tok": None,
                "decode_j_per_tok": None,
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
                "prefill_range": list(args.prefill_range),
                "cache_lengths": args.cache_lengths,
                "decode_window": args.decode_window,
                "batch_size": batch_size,
                "vision_results": resolution_payloads,
                "llm_reference_resolution": llm_resolution,
                "llm_results": {
                    "repeat": args.repeat,
                    "batch_size": batch_size,
                    "aggregate": asdict(llm_result),
                    "runs": [asdict(r) for r in llm_runs],
                    "device_time_series_runs": llm_device_time_series_runs,
                    "summary": {
                        "llm_prefill_tps": _summary(llm_prefill_tps),
                        "llm_decode_tps": _summary(llm_decode_tps),
                        "llm_ttft_ms": _summary(llm_ttft_ms),
                        "llm_decode_duration_ms": _summary(llm_decode_ms),
                        "llm_prefill_npu_latency_pct": _summary(llm_prefill_npu_latency_pct),
                        "llm_decode_npu_latency_pct": _summary(llm_decode_npu_latency_pct),
                        "avg_power_w": _summary(llm_avg_power_w),
                        "p99_power_w": _summary(llm_p99_power_w),
                        "avg_utilization_pct": _summary(llm_avg_utilization_pct),
                        "p99_utilization_pct": _summary(llm_p99_utilization_pct),
                        "avg_temperature_c": _summary(llm_avg_temperature_c),
                        "p99_temperature_c": _summary(llm_p99_temperature_c),
                        "avg_memory_used_mb": _summary(llm_avg_memory_used_mb),
                        "p99_memory_used_mb": _summary(llm_p99_memory_used_mb),
                        "total_memory_mb": _summary(llm_total_memory_mb),
                        "avg_memory_used_pct": _summary(llm_avg_memory_used_pct),
                        "p99_memory_used_pct": _summary(llm_p99_memory_used_pct),
                        "total_energy_j": _summary(llm_total_energy_j),
                    },
                },
            },
        )
        print(f"wrote: {args.json}")

    if args.csv:
        _write_csv(args.csv, csv_rows)
        print(f"wrote: {args.csv}")

    if args.plot:
        _plot_vlm_sweep(resolution_payloads=resolution_payloads, llm_result=llm_result, save_path=args.plot)
        print(f"wrote: {args.plot}")

    return 0


def add_tps_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("tps", help="Measure/sweep tokens-per-second")
    parser.epilog = (
        "Examples:\n"
        "  mblt-model-zoo tps measure --model mobilint/Llama-3.2-3B-Instruct --prefill 128 --decode 32\n"
        "  mblt-model-zoo tps measure --model <eagle3-model> --base-core-mode single --draft-core-mode global4\n"
        "  mblt-model-zoo tps measure --model <model> --input-mode file "
        "--prompt-file prompts.txt --prompt-file-strategy random "
        "--prompt-file-seed 7"
    )
    tps_sub = parser.add_subparsers(dest="tps_cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--task", default="text-generation", help="transformers pipeline task")
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
        p.add_argument("--device", default=None, help="device for pipeline (e.g., cpu, cuda:0)")
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
        p.add_argument("--base-embedding-path", default=None, help="path to custom base embedding weights")
        p.add_argument("--draft-embedding-path", default=None, help="path to custom draft embedding weights")
        p.add_argument(
            "--mxq-path",
            default=None,
            help="override mxq_path for pipeline loading (EAGLE-3 prefix options take precedence)",
        )
        for prefix in ("base", "draft", "fc"):
            p.add_argument(
                f"--{prefix}-mxq-path", default=None, help=f"override {prefix} mxq_path for pipeline loading"
            )
        p.add_argument(
            "--core-mode",
            choices=list(_CORE_MODE_CHOICES),
            default=None,
            help="NPU core mode (single, global4, global8). EAGLE-3 prefix options take precedence.",
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
        for prefix in ("base", "draft", "fc"):
            p.add_argument(
                f"--{prefix}-core-mode",
                choices=list(_CORE_MODE_CHOICES),
                default=None,
                help=f"{prefix} NPU core mode (single, global4, global8)",
            )
            p.add_argument(
                f"--{prefix}-target-cores",
                type=_parse_target_cores,
                default=None,
                help=f'{prefix} target cores (e.g., "0:0;0:1")',
            )
            p.add_argument(
                f"--{prefix}-target-clusters",
                type=_parse_target_clusters,
                default=None,
                help=f'{prefix} target clusters (e.g., "0;1")',
            )
        p.add_argument("--device-map", default=None, help="transformers device_map (optional)")
        p.add_argument("--dtype", default=None, help="dtype (e.g., auto, float16, bfloat16)")
        p.add_argument(
            "--trust-remote-code",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="pass trust_remote_code to transformers",
        )
        p.add_argument("--repeat", type=_parse_positive_int, default=1, help="number of repeated runs")
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
            "--prefill-chunk-size",
            type=_parse_positive_int_optional,
            default=None,
            help="optional prefill_chunk_size forwarded to model.generate/model.forward",
        )
        p.add_argument(
            "--batch-size",
            type=_parse_positive_int,
            default=None,
            help="batch size for synthetic inputs; defaults to model max_batch_size when available",
        )
        _add_device_tracking_args(p)
        p.set_defaults(device_backend=None)

    p_measure = tps_sub.add_parser("measure", help="Single TPS measurement")
    add_common(p_measure)
    p_measure.add_argument("--prefill", type=_parse_positive_int, default=128, help="input token count")
    p_measure.add_argument("--decode", type=_parse_positive_int, default=32, help="new tokens to generate")
    p_measure.add_argument(
        "--input-mode",
        choices=["random", "synthetic-text", "file"],
        default="random",
        help="text-only input generation mode (random|synthetic-text|file)",
    )
    p_measure.add_argument(
        "--prompt-text",
        default=None,
        help="single prompt used when --input-mode is synthetic-text",
    )
    p_measure.add_argument(
        "--prompt-file",
        default=None,
        help="text file path used when --input-mode is file",
    )
    p_measure.add_argument(
        "--prompt-file-strategy",
        choices=["first", "random"],
        default="first",
        help="non-empty line selection strategy for --input-mode file (first|random)",
    )
    p_measure.add_argument(
        "--prompt-file-seed",
        type=int,
        default=0,
        help="random seed used when --prompt-file-strategy is random",
    )
    p_measure.add_argument(
        "--image-resolution",
        type=_parse_positive_int,
        default=224,
        help="VLM only: synthetic image resolution for single measurement",
    )
    p_measure.add_argument(
        "--prompt",
        default="Describe the image in one sentence.",
        help="VLM only: fixed prompt used for synthetic image-text input",
    )
    p_measure.add_argument("--json", default=None, help="write result as JSON")
    p_measure.set_defaults(_handler=_cmd_measure)

    p_sweep = tps_sub.add_parser("sweep", help="Prefill/decode TPS sweep")
    add_common(p_sweep)
    p_sweep.add_argument(
        "--prefill-range",
        type=_parse_range,
        default=(512, 2048, 512),
        help="prefill sweep range (start:end:step)",
    )
    p_sweep.add_argument(
        "--cache-lengths",
        type=_parse_int_list,
        default=[128, 512, 1024, 2048],
        help="comma-separated cache lengths for decode sweep",
    )
    p_sweep.add_argument(
        "--decode-window",
        type=_parse_positive_int,
        default=32,
        help="decode token window measured after each cache-length prefill",
    )
    p_sweep.add_argument(
        "--image-resolutions",
        type=_parse_int_list,
        default=[224, 384, 512, 768],
        help="VLM only: comma-separated image resolutions for vision encoder sweep",
    )
    p_sweep.add_argument(
        "--llm-resolution",
        type=_parse_positive_int_optional,
        default=None,
        help="VLM only: reference resolution used for LLM benchmark (default: first image resolution)",
    )
    p_sweep.add_argument(
        "--prompt",
        default="Describe the image in one sentence.",
        help="VLM only: fixed prompt used for synthetic image-text input",
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
