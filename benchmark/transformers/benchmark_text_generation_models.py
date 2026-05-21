import argparse
import copy
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chart_utils import collect_folder_metrics, plot_scalar_chart, plot_token_chart
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

from benchmark.common.argparse_utils import parse_int_csv as _parse_int_csv_common
from benchmark.common.argparse_utils import parse_positive_int as _parse_positive_int
from benchmark.common.argparse_utils import parse_positive_int_optional as _parse_positive_int_optional
from benchmark.common.argparse_utils import parse_range_arg as _parse_range_arg
from benchmark.common.io_utils import safe_filename as _safe_filename_common
from benchmark.common.math_utils import safe_div as _safe_div
from benchmark.common.runtime_utils import clear_cuda_memory as _clear_cuda_memory
from benchmark.common.runtime_utils import cuda_device_index as _cuda_device_index
from benchmark.common.runtime_utils import is_cuda_device as _is_cuda_device
from benchmark.common.runtime_utils import is_cuda_oom_error as _is_cuda_oom_error
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
    iter_core_modes as _iter_core_modes_common,
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
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
    BenchmarkResult,
    SweepData,
    TPSMeasurer,
    npu_latency_pct,
)

_BATCH_MODE_BATCH = "batch"
_BATCH_MODE_NON_BATCH = "non_batch"
_BATCH_SWEEP_LENGTH_SCALE = 4
_SWEEP_WARMUP_PREFILL = 128
_SWEEP_WARMUP_DECODE = 32


@dataclass(frozen=True)
class TextBenchmarkTarget:
    """Resolved text-generation benchmark target with batch metadata."""

    model_id: str
    revision_candidates: list[str | None]
    label: str
    base: str
    mxq_path: str | None
    max_batch_size: int


def _safe_filename(model_id: str) -> str:
    return _safe_filename_common(model_id, replace_slash_only=True)


def _format_exception(exc: BaseException) -> str:
    """Return an exception string that remains useful for empty-message exceptions."""
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return f"{type(exc).__name__}: {exc!r}"


def _print_exception(message: str, exc: BaseException, *, debug_errors: bool) -> None:
    """Print a benchmark exception summary and optionally its traceback."""
    print(f"{message}: {_format_exception(exc)}")
    if debug_errors:
        traceback.print_exception(type(exc), exc, exc.__traceback__)


def _add_batch_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive batch target selection flags."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--batch",
        dest="batch_mode",
        action="store_const",
        const=_BATCH_MODE_BATCH,
        default=_BATCH_MODE_NON_BATCH,
        help="benchmark only targets whose config max_batch_size is greater than 1",
    )
    group.add_argument(
        "--non-batch",
        dest="batch_mode",
        action="store_const",
        const=_BATCH_MODE_NON_BATCH,
        help="benchmark only targets whose config max_batch_size is 1 (default)",
    )


def _is_gguf_model_id(model_id: str) -> bool:
    """Return whether a model id refers to a GGUF/Llama.cpp artifact."""
    return "gguf" in model_id.lower()


def _has_gguf_artifact(model_id: str, revision: str | None) -> bool:
    """Return whether a local or Hub model repository contains GGUF artifacts."""
    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        return any(path.suffix.lower() == ".gguf" for path in local_path.rglob("*"))
    if local_path.is_file():
        return local_path.suffix.lower() == ".gguf"

    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision, files_metadata=False)
    except Exception:
        return False

    siblings = getattr(info, "siblings", None) or []
    return any(str(getattr(sibling, "rfilename", "") or "").lower().endswith(".gguf") for sibling in siblings)


def _normalize_max_batch_size(value: Any) -> int | None:
    """Normalize a raw max batch size value from config metadata."""
    try:
        max_batch_size = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, max_batch_size)


def _read_raw_config(model_id: str, revision: str | None) -> dict[str, Any] | None:
    """Read raw config JSON from a local path or Hugging Face Hub/cache."""
    local_path = Path(model_id).expanduser()
    config_path = local_path / "config.json" if local_path.is_dir() else local_path
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=model_id, filename="config.json", revision=revision)
        with open(downloaded, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _extract_config_max_batch_size(payload: dict[str, Any], *, task: str) -> int | None:
    """Extract a normalized max batch size from raw model config metadata."""
    candidates: list[Any] = [payload.get("max_batch_size")]
    if task == "image-text-to-text":
        text_config = payload.get("text_config")
        vision_config = payload.get("vision_config")
        if isinstance(text_config, dict):
            candidates.append(text_config.get("max_batch_size"))
        if isinstance(vision_config, dict):
            candidates.append(vision_config.get("max_batch_size"))
    for candidate in candidates:
        max_batch_size = _normalize_max_batch_size(candidate)
        if max_batch_size is not None:
            return max_batch_size
    return None


def _resolve_config_max_batch_size(model_id: str, revision: str | None, *, task: str) -> int | None:
    """Resolve config max_batch_size for batch/non-batch target selection."""
    payload = _read_raw_config(model_id, revision)
    if payload is None:
        return None
    return _extract_config_max_batch_size(payload, task=task)


def _target_matches_batch_mode(max_batch_size: int, batch_mode: str) -> bool:
    """Return whether a resolved batch size belongs to the requested mode."""
    if batch_mode == _BATCH_MODE_BATCH:
        return max_batch_size > 1
    if batch_mode == _BATCH_MODE_NON_BATCH:
        return max_batch_size == 1
    raise ValueError(f"Unsupported batch mode: {batch_mode}")


def _target_filter_revision(model_id: str, revision_candidates: list[str | None], mxq_path: str | None) -> str | None:
    """Pick the revision used for best-effort config filtering."""
    if mxq_path:
        return revision_candidates[0] if revision_candidates else None
    return _select_revision(model_id, revision_candidates)


def _filter_text_targets_by_batch_mode(
    targets: Sequence[tuple[str, list[str | None], str, str, str | None]],
    *,
    batch_mode: str,
    task: str = "text-generation",
) -> list[TextBenchmarkTarget]:
    """Filter text-generation targets by GGUF status and config max_batch_size."""
    filtered: list[TextBenchmarkTarget] = []
    for model_id, revision_candidates, label, base, mxq_path in targets:
        revision = _target_filter_revision(model_id, revision_candidates, mxq_path)
        if _is_gguf_model_id(model_id) or _has_gguf_artifact(model_id, revision):
            print(f"Skip {label}: GGUF/Llama.cpp model is not supported by Transformers benchmark.")
            continue
        max_batch_size = _resolve_config_max_batch_size(model_id, revision, task=task)
        if max_batch_size is None:
            print(f"Skip {label}: max_batch_size is not available for batch-mode filtering.")
            continue
        if not _target_matches_batch_mode(max_batch_size, batch_mode):
            print(f"Skip {label}: max_batch_size={max_batch_size} does not match --{batch_mode.replace('_', '-')}.")
            continue
        filtered.append(
            TextBenchmarkTarget(
                model_id=model_id,
                revision_candidates=list(revision_candidates),
                label=label,
                base=base,
                mxq_path=mxq_path,
                max_batch_size=max_batch_size,
            )
        )
    return filtered


def _parse_int_list(raw: str) -> list[int]:
    return _parse_int_csv_common(raw, unique_sorted=False)


def _scale_positive_int(value: int, divisor: int) -> int:
    """Scale a positive integer down while preserving a minimum value of one."""
    return max(1, int(value) // int(divisor))


def _scale_range_arg(value: tuple[int, int, int], divisor: int) -> tuple[int, int, int]:
    """Scale a parsed range tuple down by a positive divisor."""
    start, end, step = value
    return (
        _scale_positive_int(start, divisor),
        _scale_positive_int(end, divisor),
        _scale_positive_int(step, divisor),
    )


def _scale_int_list(values: Sequence[int], divisor: int) -> list[int]:
    """Scale a sequence of positive integers down by a positive divisor."""
    return [_scale_positive_int(value, divisor) for value in values]


def _build_pipeline(
    model_id: str,
    tokenizer: str | None = None,
    revision: str | None = None,
    device: str | None = None,
    device_map: str | None = None,
    dtype: str | None = None,
    trust_remote_code: bool = True,
    core_mode: str | None = None,
    mxq_path: str | None = None,
):
    kwargs = {
        "task": "text-generation",
        "model": model_id,
        "trust_remote_code": trust_remote_code,
    }
    if device is not None:
        kwargs["device"] = device
    if revision:
        kwargs["revision"] = revision
    if tokenizer:
        kwargs["tokenizer"] = tokenizer
    if device_map:
        kwargs["device_map"] = device_map
    model_kwargs: dict[str, Any] = {}
    model_kwargs = _apply_core_mode_model_kwargs_common(model_kwargs, core_mode)
    if mxq_path:
        model_kwargs["mxq_path"] = mxq_path
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    if dtype:
        kwargs["dtype"] = dtype
        try:
            return hf_pipeline(**kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = dtype
            return hf_pipeline(**kwargs)
    return hf_pipeline(**kwargs)


def _estimate_model_weight_bytes(model_id: str, revision: str | None) -> int | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    except Exception:
        return None

    siblings = getattr(info, "siblings", None) or []
    total_bytes = 0
    for sibling in siblings:
        rfilename = getattr(sibling, "rfilename", "") or ""
        name = rfilename.lower()
        if any(ex in name for ex in ("optimizer", "training_args", "scheduler", "scaler")):
            continue
        if not name.endswith((".safetensors", ".bin", ".pt", ".pth")):
            continue
        size = getattr(sibling, "size", None)
        if isinstance(size, int) and size > 0:
            total_bytes += size
    return total_bytes if total_bytes > 0 else None


def _cuda_memory_info(device: str | None) -> tuple[int, int] | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    idx = _cuda_device_index(device)
    if idx is None:
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info(idx)
    except Exception:
        return None
    return int(free_b), int(total_b)


def _format_gib(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{float(num_bytes) / (1024**3):.2f} GiB"


def _should_precheck_cuda(args: argparse.Namespace) -> bool:
    if not args.cuda_precheck:
        return False
    if _is_cuda_device(args.device):
        return True
    # If only device_map is set (e.g. auto), target GPU topology is ambiguous.
    return False


def _normalize_repo_id(value: str) -> str:
    text = value.strip()
    if text.startswith("https://huggingface.co/"):
        text = text[len("https://huggingface.co/") :]
    return text.strip("/")


def _extract_parent_model_id(info: Any) -> str | None:
    card_data = getattr(info, "cardData", None)
    if card_data is None:
        card_data = getattr(info, "card_data", None)

    payload: dict[str, Any] | None = None
    if isinstance(card_data, dict):
        payload = card_data
    elif card_data is not None and hasattr(card_data, "to_dict"):
        try:
            payload = card_data.to_dict()
        except Exception:
            payload = None
    elif card_data is not None and hasattr(card_data, "__dict__"):
        payload = dict(card_data.__dict__)

    if not payload:
        return None

    def _pick_candidate(raw: Any) -> str | None:
        if isinstance(raw, str):
            candidate = _normalize_repo_id(raw)
            return candidate if "/" in candidate else None
        if isinstance(raw, dict):
            for key in ("model_id", "repo_id", "id", "name"):
                value = raw.get(key)
                if isinstance(value, str):
                    candidate = _normalize_repo_id(value)
                    if "/" in candidate:
                        return candidate
            return None
        if isinstance(raw, list):
            for item in raw:
                picked = _pick_candidate(item)
                if picked:
                    return picked
            return None
        return None

    for key in ("base_model", "base_models", "baseModel", "parent_model"):
        candidate = _pick_candidate(payload.get(key))
        if candidate:
            return candidate

    return None


def _resolve_original_model_ids(model_ids: Iterable[str]) -> list[str]:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
    except Exception as e:
        print(
            "Failed to initialize Hugging Face Hub API for --original-models. "
            f"Using original list_models output. Error: {e}"
        )
        return list(model_ids)

    resolved: list[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        target_id = model_id
        try:
            info = api.model_info(model_id)
            parent_id = _extract_parent_model_id(info)
            if parent_id:
                target_id = parent_id
        except Exception as e:
            print(f"Warning: failed to resolve parent model for {model_id}: {e}")

        if target_id not in seen:
            resolved.append(target_id)
            seen.add(target_id)

    return resolved


def _load_result(path: str) -> BenchmarkResult:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "benchmark" in payload and isinstance(payload["benchmark"], dict):
        payload = payload["benchmark"]
    prefill = payload.get("prefill_sweep", {})
    decode = payload.get("decode_sweep", {})
    return BenchmarkResult(
        prefill_sweep=SweepData(
            x_values=prefill.get("x_values", []),
            tps_values=prefill.get("tps_values", []),
            time_values=prefill.get("time_values", []),
            avg_total_token_latency_values=prefill.get("avg_total_token_latency_values", []),
            avg_npu_token_latency_values=prefill.get("avg_npu_token_latency_values", []),
        ),
        decode_sweep=SweepData(
            x_values=decode.get("x_values", []),
            tps_values=decode.get("tps_values", []),
            time_values=decode.get("time_values", []),
            avg_total_token_latency_values=decode.get("avg_total_token_latency_values", []),
            avg_npu_token_latency_values=decode.get("avg_npu_token_latency_values", []),
        ),
    )


def _load_device(path: str) -> dict[str, float | None] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    device = payload.get("device")
    if not isinstance(device, dict):
        return None
    out: dict[str, float | None] = {}
    for key in (
        "avg_power_w",
        "p99_power_w",
        "avg_utilization_pct",
        "p99_utilization_pct",
        "avg_temperature_c",
        "p99_temperature_c",
        "avg_memory_used_mb",
        "p99_memory_used_mb",
        "total_memory_mb",
        "avg_memory_used_pct",
        "p99_memory_used_pct",
        "total_energy_j",
        "prefill_tps_last",
        "decode_tps_last",
        "prefill_tok_per_j_last",
        "decode_tok_per_j_last",
        "prefill_j_per_tok_last",
        "decode_j_per_tok_last",
    ):
        value = device.get(key)
        out[key] = float(value) if isinstance(value, (int, float)) else None
    return out


def _aggregate_benchmark_results(results: Sequence[BenchmarkResult]) -> BenchmarkResult:
    if len(results) == 1:
        return results[0]

    def _mean_or_none(values: list[float | None]) -> float | None:
        compact = [float(v) for v in values if v is not None]
        return (sum(compact) / len(compact)) if compact else None

    def _aggregate_phase(phase: str) -> SweepData:
        first = results[0].prefill_sweep if phase == "prefill" else results[0].decode_sweep
        out = SweepData(x_values=list(first.x_values))
        for idx in range(len(first.x_values)):
            tps_values = []
            time_values = []
            total_latency_values = []
            npu_latency_values = []
            for result in results:
                src = result.prefill_sweep if phase == "prefill" else result.decode_sweep
                tps_values.append(float(src.tps_values[idx]))
                time_values.append(float(src.time_values[idx]))
                total_latency_values.append(src.avg_total_token_latency_values[idx])
                npu_latency_values.append(src.avg_npu_token_latency_values[idx])
            out.tps_values.append(sum(tps_values) / len(tps_values))
            out.time_values.append(sum(time_values) / len(time_values))
            out.avg_total_token_latency_values.append(_mean_or_none(total_latency_values))
            out.avg_npu_token_latency_values.append(_mean_or_none(npu_latency_values))
        return out

    return BenchmarkResult(prefill_sweep=_aggregate_phase("prefill"), decode_sweep=_aggregate_phase("decode"))


def _revision_exists(model_id: str, revision: str) -> bool | None:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        refs = api.list_repo_refs(model_id, repo_type="model")
        return any(branch.name == revision for branch in getattr(refs, "branches", []))
    except Exception:
        return None


def _iter_targets(
    model_ids: Iterable[str],
    *,
    revision: str | None,
    all_revisions: bool,
) -> Iterable[tuple[str, list[str | None], str, str, str | None]]:
    if not all_revisions:
        for model_id in model_ids:
            label = model_id
            base = _safe_filename(model_id)
            yield model_id, [revision], label, base, None
        return

    revision_map: list[tuple[list[str | None], str]] = [
        (["W8"], "-W8"),
        (["W4V8"], "-W4V8"),
    ]
    for model_id in model_ids:
        for revs, suffix in revision_map:
            label = f"{model_id}{suffix}"
            base = f"{_safe_filename(model_id)}{suffix}"
            yield model_id, revs, label, base, None


def _resolve_model_id_from_mxq_name(
    model_part: str,
    available_model_ids: Sequence[str],
) -> str | None:
    if model_part in available_model_ids:
        return model_part
    model_part_slash = model_part.replace("__", "/")
    if model_part_slash in available_model_ids:
        return model_part_slash

    # Fallback: match by repo basename (e.g. Qwen2.5-1.5B-Instruct).
    basename_matches = [m for m in available_model_ids if m.split("/", 1)[-1] == model_part]
    if len(basename_matches) == 1:
        return basename_matches[0]
    basename_matches_slash = [m for m in available_model_ids if m.split("/", 1)[-1] == model_part_slash]
    if len(basename_matches_slash) == 1:
        return basename_matches_slash[0]
    return None


def _iter_targets_from_mxq_dir(
    *,
    mxq_dir: Path,
    available_model_ids: Sequence[str],
) -> list[tuple[str, list[str | None], str, str, str | None]]:
    out: list[tuple[str, list[str | None], str, str, str | None]] = []
    seen_bases: set[str] = set()
    for path in sorted(mxq_dir.glob("*.mxq")):
        stem = path.stem
        if "-" not in stem:
            print(f"Skipping mxq (name format mismatch): {path.name}")
            continue
        model_part, rev_part = stem.rsplit("-", 1)
        revision = rev_part.upper()
        if revision not in ("W8", "W4V8"):
            print(f"Skipping mxq (unsupported revision suffix): {path.name}")
            continue
        resolved_model_id = _resolve_model_id_from_mxq_name(model_part, available_model_ids)
        if not resolved_model_id:
            print(
                f"Skipping mxq (cannot resolve model_id from filename): {path.name} (expected <model_id>-<W8|W4V8>.mxq)"
            )
            continue
        label = f"{resolved_model_id}-{revision}"
        base = f"{_safe_filename(resolved_model_id)}-{revision}"
        if base in seen_bases:
            print(f"Skipping mxq (duplicate target key): {path.name}")
            continue
        seen_bases.add(base)
        out.append((resolved_model_id, [revision], label, base, str(path)))
    return out


def _select_revision(
    model_id: str,
    candidates: list[str | None],
) -> str | None:
    for candidate in candidates:
        if not candidate:
            return candidate
        exists = _revision_exists(model_id, candidate)
        if exists is True:
            return candidate
        if exists is None:
            return candidate
    return None


def _build_device_tracker(args: argparse.Namespace, pipeline: Any):
    return _build_device_tracker_common(args, pipeline)


def _extract_device_metric(tracker: Any) -> dict[str, float | None]:
    return _extract_device_metric_common(tracker)


def _extract_device_time_series(tracker: Any) -> dict[str, list[dict[str, float]]]:
    return _extract_device_time_series_common(tracker)


def _weighted_two(
    a: float | None,
    a_weight: float,
    b: float | None,
    b_weight: float,
) -> float | None:
    return _weighted_two_common(a, a_weight, b, b_weight)


def _build_phase_trackers(args: argparse.Namespace, pipeline: Any) -> tuple[Any, Any]:
    return _build_phase_trackers_common(args, pipeline)


def _stop_tracker_safe(tracker: Any) -> None:
    _stop_tracker_safe_common(tracker)


def _print_device_status(args: argparse.Namespace, tracker: Any) -> None:
    _print_device_status_common(args, tracker)


def _write_device_combined_csv(path: str, rows: Sequence[dict[str, float | str | None]]) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "avg_power_w",
                "p99_power_w",
                "avg_utilization_pct",
                "p99_utilization_pct",
                "avg_temperature_c",
                "p99_temperature_c",
                "avg_memory_used_mb",
                "p99_memory_used_mb",
                "total_memory_mb",
                "avg_memory_used_pct",
                "p99_memory_used_pct",
                "total_energy_j",
                "prefill_tps_last",
                "decode_tps_last",
                "prefill_tok_per_j_last",
                "decode_tok_per_j_last",
                "prefill_j_per_tok_last",
                "decode_j_per_tok_last",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


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
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(_escape_markdown_cell(header) for header in headers) + " |\n",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |\n",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_summary_cell(value) for value in row) + " |\n")
    return "".join(lines)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows if the file exists."""
    if not path.is_file():
        return []
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _scalar_plot_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    unit_header: str,
) -> str:
    """Build a model/value table for one scalar plot."""
    table_rows = [(row.get("model"), row.get(value_key)) for row in rows]
    return _markdown_table(["Model", unit_header], table_rows)


def _token_sweep_plot_table(
    models: Sequence[str],
    metrics_by_model: Mapping[str, Any],
    *,
    value_key: str,
) -> str:
    """Build a model/token table for one token-sweep plot."""
    token_set: set[int] = set()
    for model in models:
        token_set.update(getattr(metrics_by_model[model], value_key).keys())
    tokens = sorted(token_set)
    if not tokens:
        return ""
    table_rows = []
    for model in models:
        values = getattr(metrics_by_model[model], value_key)
        table_rows.append([model, *(values.get(token) for token in tokens)])
    return _markdown_table(["Model", *(f"{token} tokens" for token in tokens)], table_rows)


def _build_text_generation_plot_tables(results_dir: Path) -> dict[str, str]:
    """Build plot-specific Markdown tables for text-generation sweep summaries."""
    metrics_by_model = collect_folder_metrics(results_dir)
    if not metrics_by_model:
        return {}
    models = sorted(metrics_by_model.keys())
    tables = {
        "prefill_tps.png": _token_sweep_plot_table(models, metrics_by_model, value_key="prefill_tps"),
        "decode_tps.png": _token_sweep_plot_table(models, metrics_by_model, value_key="decode_tps"),
    }
    rows = [{"model": model, **vars(metrics_by_model[model])} for model in models]
    scalar_specs = [
        ("prefill_tokens_per_j.png", "prefill_tokens_per_j", "tokens/J"),
        ("decode_tokens_per_j.png", "decode_tokens_per_j", "tokens/J"),
        ("avg_power_w.png", "avg_power_w", "W"),
        ("avg_temperature_c.png", "avg_temperature_c", "°C"),
        ("avg_utilization_pct.png", "avg_utilization_pct", "%"),
        ("avg_memory_used_mb.png", "avg_memory_used_mb", "MB"),
        ("total_energy_j.png", "total_energy_j", "J"),
    ]
    for filename, key, unit_header in scalar_specs:
        tables[filename] = _scalar_plot_table(rows, value_key=key, unit_header=unit_header)
    return {filename: table for filename, table in tables.items() if table}


def _build_text_generation_measure_plot_tables(results_dir: Path) -> dict[str, str]:
    """Build plot-specific Markdown tables for text-generation measure summaries."""
    rows = _read_csv_rows(results_dir / "combined_measure.csv")
    if not rows:
        return {}
    specs = [
        ("measure_prefill_tps.png", "prefill_tps_mean", "tokens/s"),
        ("measure_prefill_tokens_per_j.png", "prefill_tok_per_j_mean", "tokens/J"),
        ("measure_decode_tps.png", "decode_tps_mean", "tokens/s"),
        ("measure_decode_tokens_per_j.png", "decode_tok_per_j_mean", "tokens/J"),
        ("measure_avg_power_w.png", "avg_power_w", "W"),
        ("measure_avg_temperature_c.png", "avg_temperature_c", "°C"),
        ("measure_avg_utilization_pct.png", "avg_utilization_pct", "%"),
        ("measure_avg_memory_used_mb.png", "avg_memory_used_mb", "MB"),
        ("measure_total_energy_j.png", "total_energy_j", "J"),
    ]
    return {
        filename: table
        for filename, key, unit_header in specs
        if (table := _scalar_plot_table(rows, value_key=key, unit_header=unit_header))
    }


def _write_single_combined_markdown(
    path: str,
    tps_rows: Sequence[dict[str, Any]],
    device_rows: Sequence[dict[str, float | str | None]],
) -> None:
    if not tps_rows:
        return
    models = sorted({str(r["model"]) for r in tps_rows})
    prefill_tokens = sorted(
        {int(r["tokens"]) for r in tps_rows if str(r.get("phase")) == "prefill" and isinstance(r.get("tokens"), int)}
    )
    decode_tokens = sorted(
        {int(r["tokens"]) for r in tps_rows if str(r.get("phase")) == "decode" and isinstance(r.get("tokens"), int)}
    )
    tps_map: dict[tuple[str, str, int], float] = {}
    time_map: dict[tuple[str, str, int], float] = {}
    npu_pct_map: dict[tuple[str, str, int], float] = {}
    for row in tps_rows:
        model = str(row["model"])
        phase = str(row["phase"])
        token = int(row["tokens"])
        tps_val = row.get("tps")
        time_ms_val = row.get("time_ms")
        npu_pct_val = row.get("avg_npu_token_latency_pct")
        if isinstance(tps_val, (int, float)):
            tps_map[(model, phase, token)] = float(tps_val)
        if isinstance(time_ms_val, (int, float)):
            time_map[(model, phase, token)] = float(time_ms_val)
        if isinstance(npu_pct_val, (int, float)):
            npu_pct_map[(model, phase, token)] = float(npu_pct_val)

    device_map = {str(r["model"]): r for r in device_rows if isinstance(r.get("model"), str)}
    device_cols = [
        "avg_power_w",
        "p99_power_w",
        "avg_utilization_pct",
        "p99_utilization_pct",
        "avg_temperature_c",
        "p99_temperature_c",
        "avg_memory_used_mb",
        "p99_memory_used_mb",
        "total_memory_mb",
        "avg_memory_used_pct",
        "p99_memory_used_pct",
        "total_energy_j",
        "prefill_tps_last",
        "decode_tps_last",
        "prefill_tok_per_j_last",
        "decode_tok_per_j_last",
        "prefill_j_per_tok_last",
        "decode_j_per_tok_last",
    ]

    headers = ["model"]
    headers.extend([f"prefill_tps_{t}" for t in prefill_tokens])
    headers.extend([f"decode_tps_{t}" for t in decode_tokens])
    headers.extend([f"prefill_latency_ms_{t}" for t in prefill_tokens])
    headers.extend([f"decode_duration_ms_{t}" for t in decode_tokens])
    headers.extend([f"prefill_npu_latency_pct_{t}" for t in prefill_tokens])
    headers.extend([f"decode_npu_latency_pct_{t}" for t in decode_tokens])
    headers.extend(device_cols)

    lines = [
        "| " + " | ".join(headers) + " |\n",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |\n",
    ]
    for model in models:
        values: list[str] = [model]
        for token in prefill_tokens:
            v = tps_map.get((model, "prefill", token))
            values.append("" if v is None else f"{v:.6f}")
        for token in decode_tokens:
            v = tps_map.get((model, "decode", token))
            values.append("" if v is None else f"{v:.6f}")
        for token in prefill_tokens:
            v = time_map.get((model, "prefill", token))
            values.append("" if v is None else f"{v:.6f}")
        for token in decode_tokens:
            v = time_map.get((model, "decode", token))
            values.append("" if v is None else f"{v:.6f}")
        for token in prefill_tokens:
            v = npu_pct_map.get((model, "prefill", token))
            values.append("" if v is None else f"{v:.6f}")
        for token in decode_tokens:
            v = npu_pct_map.get((model, "decode", token))
            values.append("" if v is None else f"{v:.6f}")

        drow = device_map.get(model, {})
        for col in device_cols:
            v = drow.get(col) if isinstance(drow, dict) else None
            values.append("" if not isinstance(v, (int, float)) else f"{float(v):.6f}")
        lines.append("| " + " | ".join(values) + " |\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_text_generation_summary(results_dir: str | Path, *, measure: bool = False) -> None:
    """Write a text-generation benchmark summary Markdown with host info, plots, and table."""
    out_dir = Path(results_dir)
    table_name = "combined_measure.md" if measure else "combined.md"
    summary_name = "summary_measure.md" if measure else "summary.md"
    title = "Text Generation Measure Benchmark Summary" if measure else "Text Generation Benchmark Summary"
    prefixes = ("measure_",) if measure else None
    plot_tables = (
        _build_text_generation_measure_plot_tables(out_dir) if measure else _build_text_generation_plot_tables(out_dir)
    )
    _write_summary_markdown(
        out_dir / summary_name,
        title=title,
        host_info_path=out_dir / _HOST_PC_INFO_FILENAME,
        table_markdown_path=out_dir / table_name,
        plot_paths=_existing_png_paths(out_dir, prefixes=prefixes),
        plot_tables=plot_tables,
    )


def _rebuild_combined_outputs(results_dir: str | Path) -> None:
    out_dir = Path(results_dir)
    combined_results = []
    combined_rows = []
    combined_device_rows: list[dict[str, float | str | None]] = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("benchmark_type") == "measure" or path.name == _HOST_PC_INFO_FILENAME:
            continue
        label = payload.get("model")
        if not isinstance(label, str) or not label:
            continue
        json_path = str(path)
        result = _load_result(json_path)
        combined_results.append(result)
        combined_rows.extend(list(BenchmarkResult.iter_rows(label, result)))
        device = _load_device(json_path)
        if device:
            combined_device_rows.append(
                {
                    "model": label,
                    "avg_power_w": device.get("avg_power_w"),
                    "p99_power_w": device.get("p99_power_w"),
                    "avg_utilization_pct": device.get("avg_utilization_pct"),
                    "p99_utilization_pct": device.get("p99_utilization_pct"),
                    "avg_temperature_c": device.get("avg_temperature_c"),
                    "p99_temperature_c": device.get("p99_temperature_c"),
                    "avg_memory_used_mb": device.get("avg_memory_used_mb"),
                    "p99_memory_used_mb": device.get("p99_memory_used_mb"),
                    "total_memory_mb": device.get("total_memory_mb"),
                    "avg_memory_used_pct": device.get("avg_memory_used_pct"),
                    "p99_memory_used_pct": device.get("p99_memory_used_pct"),
                    "total_energy_j": device.get("total_energy_j"),
                    "prefill_tps_last": device.get("prefill_tps_last"),
                    "decode_tps_last": device.get("decode_tps_last"),
                    "prefill_tok_per_j_last": device.get("prefill_tok_per_j_last"),
                    "decode_tok_per_j_last": device.get("decode_tok_per_j_last"),
                    "prefill_j_per_tok_last": device.get("prefill_j_per_tok_last"),
                    "decode_j_per_tok_last": device.get("decode_j_per_tok_last"),
                }
            )

    if not combined_results:
        print("No existing JSON results matched the current target set. Nothing to aggregate.")
        _write_text_generation_summary(out_dir)
        return

    combined_csv = os.path.join(out_dir, "combined.csv")
    combined_md = os.path.join(out_dir, "combined.md")
    BenchmarkResult.write_combined_csv(combined_csv, combined_rows)
    _write_single_combined_markdown(
        combined_md,
        tps_rows=combined_rows,
        device_rows=combined_device_rows,
    )

    folder_metrics = collect_folder_metrics(out_dir)
    if folder_metrics:
        models = sorted(folder_metrics.keys())
        labels = ["benchmark"]
        metrics_by_folder = [folder_metrics]

        plot_token_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            token_selector=lambda m: m.prefill_tps,
            title="Prefill Tokens Per Second",
            x_label="Tokens Per Second",
            output_path=out_dir / "prefill_tps.png",
        )
        scalar_specs = [
            (
                "prefill_tokens_per_j.png",
                "Prefill Tokens Per Joule",
                "Tokens Per Joule",
                lambda m: m.prefill_tokens_per_j,
            ),
        ]
        plot_token_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            token_selector=lambda m: m.decode_tps,
            title="Decode Tokens Per Second",
            x_label="Tokens Per Second",
            output_path=out_dir / "decode_tps.png",
        )
        scalar_specs.extend(
            [
                (
                    "decode_tokens_per_j.png",
                    "Decode Tokens Per Joule",
                    "Tokens Per Joule",
                    lambda m: m.decode_tokens_per_j,
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
        )
        for filename, title, x_label, selector in scalar_specs:
            plot_scalar_chart(
                models=models,
                folder_labels=labels,
                metrics_by_folder=metrics_by_folder,
                scalar_selector=selector,
                title=title,
                x_label=x_label,
                output_path=out_dir / filename,
            )

    if combined_device_rows:
        device_csv = os.path.join(out_dir, "combined_device.csv")
        _write_device_combined_csv(device_csv, combined_device_rows)

    _write_text_generation_summary(out_dir)


def _add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by text-generation benchmark subcommands."""
    _add_pipeline_device_args(parser, device_default=None, trust_remote_code_default=True)
    _add_batch_selection_args(parser)
    parser.add_argument("--model", default=None, help="single model id to benchmark (optional)")
    parser.add_argument("--tokenizer", default=None, help="tokenizer id or local path (optional)")
    parser.add_argument("--revision", default=None, help="model revision (e.g., W8)")
    parser.add_argument("--mxq-path", default=None, help="override mxq_path for pipeline loading")
    parser.add_argument("--all", action="store_true", help="benchmark W8 and W4V8 revisions only (skip main)")
    parser.add_argument(
        "--mxq-dir",
        default=None,
        help=(
            "directory containing local mxq files. "
            "When set, only files matching <model_id>-<W8|W4V8>.mxq are benchmarked."
        ),
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=_parse_positive_int_optional,
        default=None,
        help="optional prefill_chunk_size forwarded to model.generate/model.forward",
    )
    parser.add_argument(
        "--core-mode",
        choices=[*list(_CORE_MODE_CHOICES_COMMON), "all"],
        default="global8",
        help="core mode passed to model_kwargs; all expands to single/global4/global8 (default: global8)",
    )
    parser.add_argument("--repeat", type=_parse_positive_int, default=1, help="number of repeated measured runs")
    parser.add_argument("--skip-existing", action="store_true", help="skip models with existing outputs")
    parser.add_argument(
        "--rebuild-charts",
        action="store_true",
        help="skip benchmarking and rebuild combined outputs from existing JSON files",
    )
    parser.add_argument(
        "--warmup",
        type=_parse_positive_int,
        default=1,
        help="number of warmup runs before measured run",
    )
    parser.add_argument(
        "--original-models",
        action="store_true",
        help="resolve each Mobilint model to its parent/base model from HF Hub and benchmark unique parent ids",
    )
    _add_device_tracking_args(parser)
    parser.add_argument(
        "--cuda-precheck",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="best-effort CUDA VRAM pre-check before loading each model (default: on)",
    )
    parser.add_argument(
        "--cuda-precheck-margin",
        type=float,
        default=1.15,
        help="required free VRAM factor versus estimated model weights (default: 1.15)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="output directory (default: benchmark/transformers/results/text_generation)",
    )
    parser.add_argument(
        "--debug-errors",
        action="store_true",
        help="print full tracebacks for per-target benchmark failures",
    )


def _add_measure_args(parser: argparse.ArgumentParser) -> None:
    """Add TPS measure-aligned fixed measurement arguments."""
    parser.add_argument("--prefill", type=_parse_positive_int, default=128, help="prefill token count (default: 128)")
    parser.add_argument("--decode", type=_parse_positive_int, default=32, help="decode token count (default: 32)")


def _add_sweep_args(parser: argparse.ArgumentParser) -> None:
    """Add TPS sweep-aligned grid measurement arguments."""
    parser.add_argument(
        "--prefill-range",
        type=_parse_range_arg,
        default=(512, 2048, 512),
        help="prefill sweep range as 'start:end:step' (default: 512:2048:512)",
    )
    parser.add_argument(
        "--cache-lengths",
        type=_parse_int_list,
        default=[128, 512, 1024, 2048],
        help="decode sweep cache lengths as comma-separated integers (default: 128,512,1024,2048)",
    )
    parser.add_argument(
        "--decode-window",
        type=_parse_positive_int,
        default=32,
        help="decode token window measured after each cache-length prefill (default: 32)",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the text-generation benchmark argument parser."""
    parser = argparse.ArgumentParser(description="Benchmark text-generation models.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure", help="measure fixed prefill/decode TPS")
    _add_common_benchmark_args(measure)
    _add_measure_args(measure)
    measure.set_defaults(_handler=_run_measure)
    sweep = subparsers.add_parser("sweep", help="run prefill/decode TPS sweeps")
    _add_common_benchmark_args(sweep)
    _add_sweep_args(sweep)
    sweep.set_defaults(_handler=_run_sweep)
    return parser


def _flag_present(raw_argv: Sequence[str], flag: str) -> bool:
    """Return whether a flag appears in raw argv."""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in raw_argv)


def _resolve_batch_core_mode(args: argparse.Namespace, *, core_mode_explicit: bool) -> None:
    """Apply the single-core constraint for batch LLM benchmarks."""
    if args.batch_mode != _BATCH_MODE_BATCH:
        return
    if core_mode_explicit and args.core_mode != "single":
        raise SystemExit("--batch only supports --core-mode single for batch LLM benchmarks.")
    args.core_mode = "single"


def _resolve_batch_sweep_lengths(args: argparse.Namespace, raw_argv: Sequence[str]) -> None:
    """Scale default sweep lengths down for batch text-generation benchmarks."""
    if args.command != "sweep" or args.batch_mode != _BATCH_MODE_BATCH:
        return

    scaled: list[str] = []
    if not _flag_present(raw_argv, "--prefill-range"):
        original_prefill_range = args.prefill_range
        args.prefill_range = _scale_range_arg(args.prefill_range, _BATCH_SWEEP_LENGTH_SCALE)
        scaled.append(f"prefill_range={original_prefill_range}->{args.prefill_range}")
    if not _flag_present(raw_argv, "--cache-lengths"):
        original_cache_lengths = list(args.cache_lengths)
        args.cache_lengths = _scale_int_list(args.cache_lengths, _BATCH_SWEEP_LENGTH_SCALE)
        scaled.append(f"cache_lengths={original_cache_lengths}->{args.cache_lengths}")

    if scaled:
        print(
            "Auto-scaled --batch sweep lengths by "
            f"1/{_BATCH_SWEEP_LENGTH_SCALE}; decode_window remains {args.decode_window}: "
            + ", ".join(scaled)
        )


def _resolve_runtime_defaults(args: argparse.Namespace, raw_argv: Sequence[str]) -> None:
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
    args.device_backend = _resolve_default_device_backend_common(
        device_backend=args.device_backend,
        device_backend_explicit=device_backend_explicit,
        model_id=args.model,
        mxq_path=args.mxq_path,
        mxq_dir=args.mxq_dir,
        original_models=args.original_models,
    )
    if not device_explicit:
        print(f"Auto-set --device={args.device}")
    if not device_backend_explicit:
        if args.model or args.mxq_path or args.mxq_dir:
            print(f"Auto-set --device-backend={args.device_backend} (based on target/device policy)")
        else:
            print("Auto-set --device-backend per target (based on target/device policy)")
    _resolve_batch_core_mode(args, core_mode_explicit=core_mode_explicit)
    _resolve_batch_sweep_lengths(args, raw_argv)


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
    """Run the selected text-generation benchmark subcommand."""
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _resolve_runtime_defaults(args, raw_argv)
    return args._handler(args)


def _resolve_text_generation_results_dir(args: argparse.Namespace) -> str:
    """Resolve and create the text-generation benchmark results directory."""
    results_dir = str(
        Path(args.results_dir).resolve()
        if args.results_dir
        else Path(__file__).resolve().parent / "results" / "text_generation"
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _run_sweep(args: argparse.Namespace) -> int:
    """Run multi-model text-generation sweep benchmarks."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    results_dir = _resolve_text_generation_results_dir(args)
    if args.rebuild_charts:
        print("Rebuilding combined outputs from existing JSON files only...")
        _rebuild_combined_outputs(results_dir)
        return 0

    disable_npu_specific_args = bool(args.original_models and not args.mxq_dir)
    if disable_npu_specific_args:
        print("Note: --original-models is enabled; skipping NPU-specific parameters (core_mode/prefill_chunk_size).")
    _resolve_batch_core_mode(args, core_mode_explicit=bool(getattr(args, "_core_mode_explicit", False)))

    available = list_models(tasks="text-generation")
    available_model_ids = available.get("text-generation", [])
    if not available_model_ids:
        print("No text-generation models found.")
        return 0

    _collect_host_pc_info(results_dir)

    if args.mxq_dir:
        mxq_dir = Path(args.mxq_dir).expanduser().resolve()
        if not mxq_dir.is_dir():
            raise SystemExit(f"--mxq-dir is not a directory: {mxq_dir}")
        if args.model or args.original_models or args.all or args.revision or args.mxq_path:
            print(
                "Note: --mxq-dir is set, so --model/--original-models/--all/--revision/--mxq-path are ignored "
                "(revision is taken from mxq filename suffix)."
            )
        targets = _iter_targets_from_mxq_dir(
            mxq_dir=mxq_dir,
            available_model_ids=available_model_ids,
        )
        if not targets:
            raise SystemExit("No valid mxq targets found. Expected files named <model_id>-<W8|W4V8>.mxq in --mxq-dir.")
        print(f"Using local mxq targets from {mxq_dir}: {len(targets)} files")
    else:
        model_ids = [str(args.model)] if args.model else available_model_ids
        if args.original_models:
            original_count = len(model_ids)
            model_ids = _resolve_original_model_ids(model_ids)
            print(
                f"Using parent/original model ids: {len(model_ids)} unique models "
                f"(from {original_count} listed models)."
            )
            if args.all or args.revision:
                print(
                    "Note: --all/--revision are applied to resolved original model ids "
                    "(requested revisions may not exist)."
                )
        targets = list(_iter_targets(model_ids, revision=args.revision, all_revisions=args.all))
        if args.mxq_path:
            targets = [
                (model_id, revisions, label, base, args.mxq_path) for model_id, revisions, label, base, _ in targets
            ]
    filtered_targets = _filter_text_targets_by_batch_mode(targets, batch_mode=args.batch_mode)
    core_modes = [None] if disable_npu_specific_args else _iter_core_modes_common(args.core_mode)
    run_targets: list[tuple[str, list[str | None], str, str, str | None, str | None, int]] = []
    for target in filtered_targets:
        for core_mode in core_modes:
            mode_label, mode_base = _append_core_mode_suffix_common(target.label, target.base, core_mode)
            run_targets.append(
                (
                    target.model_id,
                    target.revision_candidates,
                    mode_label,
                    mode_base,
                    target.mxq_path,
                    core_mode,
                    target.max_batch_size,
                )
            )


    for model_id, revision_candidates, label, base, mxq_path, core_mode, batch_size in tqdm(
        run_targets,
        desc="Benchmarking models",
        total=len(run_targets),
        unit="model-mode",
    ):
        target_args = _args_for_target_device_backend(args, model_id=model_id, mxq_path=mxq_path)
        # Ensure pre-check sees memory state after releasing previous model.
        if _is_cuda_device(args.device):
            _clear_cuda_memory(args.device)
        print(f"=== {label} ===")
        if mxq_path:
            print(f"Using local mxq: {mxq_path}")
        if mxq_path:
            revision = revision_candidates[0] if revision_candidates else None
        else:
            revision = _select_revision(model_id, revision_candidates)
        if args.all and not args.mxq_dir and revision is None:
            print("Skipping (missing revisions).")
            continue
        json_path = os.path.join(results_dir, f"{base}.json")
        png_path = os.path.join(results_dir, f"{base}.png")
        if args.skip_existing and os.path.isfile(json_path) and os.path.isfile(png_path):
            print("Skipping (results exist).")
            continue
        print(
            "Run config: "
            f"batch_size={batch_size} core_mode={core_mode or 'default'} revision={revision or 'main'} "
            f"device={args.device} device_backend={target_args.device_backend} "
            f"prefill_chunk_size={args.prefill_chunk_size if args.prefill_chunk_size is not None else 'auto'}"
        )
        print(
            "Sweep config: "
            f"warmup_prefill={_SWEEP_WARMUP_PREFILL} warmup_decode={_SWEEP_WARMUP_DECODE} "
            f"prefill_range={args.prefill_range} cache_lengths={args.cache_lengths} "
            f"decode_window={args.decode_window} repeat={args.repeat} warmup={args.warmup}"
        )
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
            try:
                pipeline = _build_pipeline(
                    model_id,
                    tokenizer=args.tokenizer,
                    revision=revision,
                    device=args.device,
                    device_map=args.device_map,
                    dtype=args.dtype,
                    trust_remote_code=args.trust_remote_code,
                    core_mode=core_mode,
                    mxq_path=mxq_path,
                )
            except Exception as e:
                if _is_cuda_oom_error(e):
                    print(f"Skipping (CUDA OOM while loading model): {e}")
                    _clear_cuda_memory(args.device)
                    continue
                if args.all and not args.mxq_dir and _revision_exists(model_id, revision or "") is None:
                    _print_exception(
                        f"Skipping (failed to load revision {revision})",
                        e,
                        debug_errors=args.debug_errors,
                    )
                else:
                    _print_exception("Skipping (failed to load model)", e, debug_errors=args.debug_errors)
                continue

            measurer = TPSMeasurer(pipeline)
            tracker_prefill, tracker_decode = _build_phase_trackers(target_args, pipeline)
            _print_device_status(target_args, tracker_prefill)
            resolved_prefill_chunk_size = None if disable_npu_specific_args else args.prefill_chunk_size
            for i in tqdm(range(args.warmup), desc=f"{label} warmup", leave=False):
                measurer.measure(
                    num_prefill=_SWEEP_WARMUP_PREFILL,
                    num_decode=_SWEEP_WARMUP_DECODE,
                    batch_size=batch_size,
                    prefill_chunk_size=resolved_prefill_chunk_size,
                    trace_path=None,
                    show_progress=True,
                    progress_desc=f"{label} warmup generate {i + 1}/{args.warmup}",
                )
            run_results: list[BenchmarkResult] = []
            for repeat_idx in tqdm(range(args.repeat), desc=f"{label} measured runs", leave=False):
                try:
                    run_results.append(
                        measurer.measure_full(
                            prefill_range=args.prefill_range,
                            cache_lengths=args.cache_lengths,
                            decode_window=args.decode_window,
                            batch_size=batch_size,
                            prefill_chunk_size=resolved_prefill_chunk_size,
                            show_progress=True,
                            progress_prefix=f"{label} run {repeat_idx + 1}/{args.repeat}",
                            on_prefill_start=(lambda: tracker_prefill.start()) if tracker_prefill is not None else None,
                            on_prefill_end=(lambda: tracker_prefill.stop()) if tracker_prefill is not None else None,
                            on_decode_start=(lambda: tracker_decode.start()) if tracker_decode is not None else None,
                            on_decode_end=(lambda: tracker_decode.stop()) if tracker_decode is not None else None,
                        )
                    )
                finally:
                    _stop_tracker_safe(tracker_prefill)
                    _stop_tracker_safe(tracker_decode)
            result = _aggregate_benchmark_results(run_results)
        except Exception as e:
            if _is_cuda_oom_error(e):
                print(f"Skipping (CUDA OOM during benchmark): {e}")
                _release_pipeline(pipeline, args.device)
                continue
            _print_exception("Skipping (benchmark failed)", e, debug_errors=args.debug_errors)
            _release_pipeline(pipeline, args.device)
            continue

        if result.prefill_sweep.avg_total_token_latency_values:
            avg_total = result.prefill_sweep.avg_total_token_latency_values[-1]
            avg_npu = result.prefill_sweep.avg_npu_token_latency_values[-1]
            avg_npu_str = f"{avg_npu * 1000.0:.3f}ms" if avg_npu is not None else "n/a"
            npu_pct = npu_latency_pct(avg_total, avg_npu)
            npu_pct_str = f"{npu_pct:.1f}%" if npu_pct is not None else "n/a"
            print(
                "Avg prefill token latency (last): "
                f"total={avg_total * 1000.0:.3f}ms npu={avg_npu_str} npu_pct={npu_pct_str}"
            )
        if result.decode_sweep.avg_total_token_latency_values:
            avg_total = result.decode_sweep.avg_total_token_latency_values[-1]
            avg_npu = result.decode_sweep.avg_npu_token_latency_values[-1]
            avg_npu_str = f"{avg_npu * 1000.0:.3f}ms" if avg_npu is not None else "n/a"
            npu_pct = npu_latency_pct(avg_total, avg_npu)
            npu_pct_str = f"{npu_pct:.1f}%" if npu_pct is not None else "n/a"
            print(
                "Avg decode token latency (last): "
                f"total={avg_total * 1000.0:.3f}ms npu={avg_npu_str} npu_pct={npu_pct_str}"
            )
        device_payload: dict[str, Any] | None = None
        device_time_series_payload: dict[str, dict[str, list[dict[str, float]]]] | None = None
        if tracker_prefill is not None and tracker_decode is not None:
            prefill_metric = _extract_device_metric(tracker_prefill)
            decode_metric = _extract_device_metric(tracker_decode)
            device_time_series_payload = {
                "prefill": _extract_device_time_series(tracker_prefill),
                "decode": _extract_device_time_series(tracker_decode),
            }
            prefill_phase_duration_s = float(getattr(result, "prefill_phase_duration_s", 0.0) or 0.0)
            decode_phase_duration_s = float(getattr(result, "decode_phase_duration_s", 0.0) or 0.0)
            prefill_avg_power = prefill_metric.get("avg_power_w")
            decode_avg_power = decode_metric.get("avg_power_w")
            prefill_energy = prefill_avg_power * prefill_phase_duration_s if prefill_avg_power is not None else None
            decode_energy = decode_avg_power * decode_phase_duration_s if decode_avg_power is not None else None
            total_energy = None
            if prefill_energy is not None and decode_energy is not None:
                total_energy = prefill_energy + decode_energy
            avg_power = _weighted_two(
                prefill_metric.get("avg_power_w"),
                prefill_phase_duration_s,
                decode_metric.get("avg_power_w"),
                decode_phase_duration_s,
            )
            p99_power = max(
                [v for v in (prefill_metric.get("p99_power_w"), decode_metric.get("p99_power_w")) if v is not None],
                default=None,
            )
            avg_utilization = _weighted_two(
                prefill_metric.get("avg_utilization_pct"),
                prefill_phase_duration_s,
                decode_metric.get("avg_utilization_pct"),
                decode_phase_duration_s,
            )
            p99_utilization = max(
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
            avg_temperature = _weighted_two(
                prefill_metric.get("avg_temperature_c"),
                prefill_phase_duration_s,
                decode_metric.get("avg_temperature_c"),
                decode_phase_duration_s,
            )
            p99_temperature = max(
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
            avg_memory_used_mb = _weighted_two(
                prefill_metric.get("avg_memory_used_mb"),
                prefill_phase_duration_s,
                decode_metric.get("avg_memory_used_mb"),
                decode_phase_duration_s,
            )
            p99_memory_used_mb = max(
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
            total_memory_mb = max(
                [
                    v
                    for v in (prefill_metric.get("total_memory_mb"), decode_metric.get("total_memory_mb"))
                    if v is not None
                ],
                default=None,
            )
            avg_memory_used_pct = _weighted_two(
                prefill_metric.get("avg_memory_used_pct"),
                prefill_phase_duration_s,
                decode_metric.get("avg_memory_used_pct"),
                decode_phase_duration_s,
            )
            p99_memory_used_pct = max(
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
            prefill_last = float(result.prefill_sweep.tps_values[-1]) if result.prefill_sweep.tps_values else None
            decode_last = float(result.decode_sweep.tps_values[-1]) if result.decode_sweep.tps_values else None
            prefill_tpj = (
                _safe_div(prefill_last, prefill_avg_power)
                if prefill_last is not None and prefill_avg_power is not None
                else None
            )
            decode_tpj = (
                _safe_div(decode_last, decode_avg_power)
                if decode_last is not None and decode_avg_power is not None
                else None
            )
            device_payload = {
                "avg_power_w": avg_power,
                "p99_power_w": p99_power,
                "avg_utilization_pct": avg_utilization,
                "p99_utilization_pct": p99_utilization,
                "avg_temperature_c": avg_temperature,
                "p99_temperature_c": p99_temperature,
                "avg_memory_used_mb": avg_memory_used_mb,
                "p99_memory_used_mb": p99_memory_used_mb,
                "total_memory_mb": total_memory_mb,
                "avg_memory_used_pct": avg_memory_used_pct,
                "p99_memory_used_pct": p99_memory_used_pct,
                "total_energy_j": total_energy,
                "prefill_tps_last": prefill_last,
                "decode_tps_last": decode_last,
                "prefill_tok_per_j_last": prefill_tpj,
                "decode_tok_per_j_last": decode_tpj,
                "prefill_j_per_tok_last": _safe_div(1.0, prefill_tpj) if prefill_tpj else None,
                "decode_j_per_tok_last": _safe_div(1.0, decode_tpj) if decode_tpj else None,
                "prefill_avg_power_w": prefill_metric.get("avg_power_w"),
                "prefill_p99_power_w": prefill_metric.get("p99_power_w"),
                "prefill_avg_utilization_pct": prefill_metric.get("avg_utilization_pct"),
                "prefill_p99_utilization_pct": prefill_metric.get("p99_utilization_pct"),
                "prefill_avg_temperature_c": prefill_metric.get("avg_temperature_c"),
                "prefill_p99_temperature_c": prefill_metric.get("p99_temperature_c"),
                "prefill_avg_memory_used_mb": prefill_metric.get("avg_memory_used_mb"),
                "prefill_p99_memory_used_mb": prefill_metric.get("p99_memory_used_mb"),
                "prefill_avg_memory_used_pct": prefill_metric.get("avg_memory_used_pct"),
                "prefill_p99_memory_used_pct": prefill_metric.get("p99_memory_used_pct"),
                "decode_avg_power_w": decode_metric.get("avg_power_w"),
                "decode_p99_power_w": decode_metric.get("p99_power_w"),
                "decode_avg_utilization_pct": decode_metric.get("avg_utilization_pct"),
                "decode_p99_utilization_pct": decode_metric.get("p99_utilization_pct"),
                "decode_avg_temperature_c": decode_metric.get("avg_temperature_c"),
                "decode_p99_temperature_c": decode_metric.get("p99_temperature_c"),
                "decode_avg_memory_used_mb": decode_metric.get("avg_memory_used_mb"),
                "decode_p99_memory_used_mb": decode_metric.get("p99_memory_used_mb"),
                "decode_avg_memory_used_pct": decode_metric.get("avg_memory_used_pct"),
                "decode_p99_memory_used_pct": decode_metric.get("p99_memory_used_pct"),
                "prefill_energy_j": prefill_energy,
                "decode_energy_j": decode_energy,
                "prefill_phase_duration_s": prefill_phase_duration_s,
                "decode_phase_duration_s": decode_phase_duration_s,
            }
            print(
                "Power/Efficiency: "
                f"avg_power={avg_power if avg_power is not None else 'n/a'}W "
                f"avg_util={avg_utilization if avg_utilization is not None else 'n/a'}% "
                f"avg_mem_used={avg_memory_used_mb if avg_memory_used_mb is not None else 'n/a'}MB "
                f"prefill_avg_power={prefill_avg_power if prefill_avg_power is not None else 'n/a'}W "
                f"decode_avg_power={decode_avg_power if decode_avg_power is not None else 'n/a'}W "
                f"prefill_tok_per_j(last)={prefill_tpj if prefill_tpj is not None else 'n/a'} "
                f"decode_tok_per_j(last)={decode_tpj if decode_tpj is not None else 'n/a'}"
            )

        payload: dict[str, Any] = {
            "model": label,
            "batch_mode": args.batch_mode,
            "batch_size": batch_size,
            "benchmark": asdict(result),
            "device": device_payload,
            "device_time_series": device_time_series_payload,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        measurer.plot_and_save(result, save_path=png_path)

        _release_pipeline(pipeline, args.device)

    _rebuild_combined_outputs(results_dir)

    return 0


def _summary(values: Sequence[float]) -> dict[str, float]:
    """Return common summary statistics for measured scalar values."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    def _percentile(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * q
        lo = int(idx)
        hi = min(lo + 1, len(vals) - 1)
        frac = idx - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "mean": sum(vals) / len(vals),
        "min": vals[0],
        "max": vals[-1],
        "p50": _percentile(0.50),
        "p95": _percentile(0.95),
        "p99": _percentile(0.99),
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Return a mean value, or None when no values are present."""
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else None


def _measure_device_payload(runs: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Build an aggregate device payload from measured run dictionaries."""
    device_keys = (
        "avg_power_w",
        "p99_power_w",
        "avg_utilization_pct",
        "p99_utilization_pct",
        "avg_temperature_c",
        "p99_temperature_c",
        "avg_memory_used_mb",
        "p99_memory_used_mb",
        "total_memory_mb",
        "avg_memory_used_pct",
        "p99_memory_used_pct",
        "total_energy_j",
        "prefill_tokens_per_j",
        "decode_tokens_per_j",
        "prefill_j_per_token",
        "decode_j_per_token",
    )
    payload: dict[str, Any] = {}
    for key in device_keys:
        vals = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]
        if key.startswith("p99_") or key == "total_memory_mb":
            payload[key] = max(vals) if vals else None
        elif key == "total_energy_j":
            payload[key] = sum(vals) if vals else None
        else:
            payload[key] = _mean_or_none(vals)
    payload["prefill_tps_last"] = runs[-1].get("prefill_tps") if runs else None
    payload["decode_tps_last"] = runs[-1].get("decode_tps") if runs else None
    return payload if any(v is not None for v in payload.values()) else None


def _collect_measure_rows(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert per-model measure payloads to combined summary rows."""
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        summary = payload.get("summary", {})
        device = payload.get("device") or {}
        rows.append(
            {
                "model": payload.get("model"),
                "batch_mode": payload.get("batch_mode"),
                "batch_size": payload.get("batch_size"),
                "prefill_tokens": payload.get("prefill"),
                "decode_tokens": payload.get("decode"),
                "repeat": payload.get("repeat"),
                "prefill_tps_mean": summary.get("prefill_tps", {}).get("mean"),
                "decode_tps_mean": summary.get("decode_tps", {}).get("mean"),
                "ttft_ms_mean": summary.get("ttft_ms", {}).get("mean"),
                "decode_duration_ms_mean": summary.get("decode_duration_ms", {}).get("mean"),
                "total_time_ms_mean": summary.get("total_time_ms", {}).get("mean"),
                "prefill_npu_latency_pct_mean": summary.get("prefill_npu_latency_pct", {}).get("mean"),
                "decode_npu_latency_pct_mean": summary.get("decode_npu_latency_pct", {}).get("mean"),
                "avg_power_w": device.get("avg_power_w"),
                "p99_power_w": device.get("p99_power_w"),
                "avg_utilization_pct": device.get("avg_utilization_pct"),
                "avg_temperature_c": device.get("avg_temperature_c"),
                "avg_memory_used_mb": device.get("avg_memory_used_mb"),
                "total_energy_j": device.get("total_energy_j"),
                "prefill_tok_per_j_mean": device.get("prefill_tokens_per_j"),
                "decode_tok_per_j_mean": device.get("decode_tokens_per_j"),
            }
        )
    return rows


def _write_measure_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write combined measure rows as a Markdown table."""
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |\n", "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |\n"]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(h) is None else str(row.get(h)) for h in headers) + " |\n")
    path.write_text("".join(lines), encoding="utf-8")


def _plot_measure_charts(results_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Create combined measure bar charts."""
    if not rows:
        return
    import matplotlib.pyplot as plt

    models = [str(row.get("model")) for row in rows]
    specs = [
        ("measure_prefill_tps.png", "prefill_tps_mean", "Prefill Tokens Per Second", "Tokens Per Second"),
        (
            "measure_prefill_tokens_per_j.png",
            "prefill_tok_per_j_mean",
            "Prefill Tokens Per Joule",
            "Tokens Per Joule",
        ),
        ("measure_decode_tps.png", "decode_tps_mean", "Decode Tokens Per Second", "Tokens Per Second"),
        (
            "measure_decode_tokens_per_j.png",
            "decode_tok_per_j_mean",
            "Decode Tokens Per Joule",
            "Tokens Per Joule",
        ),
        ("measure_avg_power_w.png", "avg_power_w", "Power", "Power (Watts)"),
        ("measure_avg_temperature_c.png", "avg_temperature_c", "Temperature", "Temperature (Celsius)"),
        ("measure_avg_utilization_pct.png", "avg_utilization_pct", "Utilization", "Utilization (Percent)"),
        ("measure_avg_memory_used_mb.png", "avg_memory_used_mb", "Memory Used Megabytes", "Memory Used (Megabytes)"),
        ("measure_total_energy_j.png", "total_energy_j", "Total Energy", "Energy (Joules)"),
    ]
    for filename, key, title, xlabel in specs:
        values = [float(row.get(key) or 0.0) for row in rows]
        height = max(4.0, 0.35 * len(models) + 1.5)
        fig, ax = plt.subplots(figsize=(10, height))
        ax.barh(models, values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=220)
        plt.close(fig)


def _rebuild_measure_outputs(results_dir: str | Path) -> None:
    """Rebuild combined text-generation measure CSV, Markdown, and charts."""
    out_dir = Path(results_dir)
    payloads: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*_measure.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("benchmark_type") == "measure":
            payloads.append(payload)
    if not payloads:
        print("No measure JSON results found. Nothing to aggregate.")
        _write_text_generation_summary(out_dir, measure=True)
        return
    rows = _collect_measure_rows(payloads)
    import csv

    with (out_dir / "combined_measure.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_measure_markdown(out_dir / "combined_measure.md", rows)
    _plot_measure_charts(out_dir, rows)
    _write_text_generation_summary(out_dir, measure=True)


def _collect_text_run_targets(
    args: argparse.Namespace,
) -> tuple[str, bool, list[tuple[str, list[str | None], str, str, str | None, str | None, int]]]:
    """Resolve text-generation benchmark targets and core-mode expansion."""
    available = list_models(tasks="text-generation")
    available_model_ids = available.get("text-generation", [])
    if not available_model_ids:
        print("No text-generation models found.")
        return "", False, []
    results_dir = str(
        Path(args.results_dir).resolve()
        if args.results_dir
        else Path(__file__).resolve().parent / "results" / "text_generation"
    )
    os.makedirs(results_dir, exist_ok=True)
    disable_npu_specific_args = bool(args.original_models and not args.mxq_dir)
    _resolve_batch_core_mode(args, core_mode_explicit=bool(getattr(args, "_core_mode_explicit", False)))
    targets: list[tuple[str, list[str | None], str, str, str | None]]
    if args.mxq_dir:
        mxq_dir = Path(args.mxq_dir).expanduser().resolve()
        if not mxq_dir.is_dir():
            raise SystemExit(f"--mxq-dir is not a directory: {mxq_dir}")
        targets = _iter_targets_from_mxq_dir(mxq_dir=mxq_dir, available_model_ids=available_model_ids)
        if not targets:
            raise SystemExit("No valid mxq targets found. Expected files named <model_id>-<W8|W4V8>.mxq in --mxq-dir.")
    else:
        model_ids = [str(args.model)] if args.model else available_model_ids
        if args.original_models:
            model_ids = _resolve_original_model_ids(model_ids)
        targets = list(_iter_targets(model_ids, revision=args.revision, all_revisions=args.all))
        if args.mxq_path:
            targets = [
                (model_id, revisions, label, base, args.mxq_path) for model_id, revisions, label, base, _ in targets
            ]
    filtered_targets = _filter_text_targets_by_batch_mode(targets, batch_mode=args.batch_mode)
    core_modes = [None] if disable_npu_specific_args else _iter_core_modes_common(args.core_mode)
    run_targets: list[tuple[str, list[str | None], str, str, str | None, str | None, int]] = []
    for target in filtered_targets:
        for core_mode in core_modes:
            mode_label, mode_base = _append_core_mode_suffix_common(target.label, target.base, core_mode)
            run_targets.append(
                (
                    target.model_id,
                    target.revision_candidates,
                    mode_label,
                    mode_base,
                    target.mxq_path,
                    core_mode,
                    target.max_batch_size,
                )
            )
    return results_dir, disable_npu_specific_args, run_targets


def _run_measure(args: argparse.Namespace) -> int:
    """Run multi-model text-generation fixed prefill/decode benchmarks."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.rebuild_charts:
        _rebuild_measure_outputs(_resolve_text_generation_results_dir(args))
        return 0
    results_dir, disable_npu_specific_args, run_targets = _collect_text_run_targets(args)
    if not run_targets:
        return 0
    if disable_npu_specific_args:
        print("Note: --original-models is enabled; skipping NPU-specific parameters (core_mode/prefill_chunk_size).")
    _collect_host_pc_info(results_dir)
    for model_id, revision_candidates, label, base, mxq_path, core_mode, batch_size in tqdm(
        run_targets, desc="Measuring models", total=len(run_targets), unit="model-mode"
    ):
        target_args = _args_for_target_device_backend(args, model_id=model_id, mxq_path=mxq_path)
        if _is_cuda_device(args.device):
            _clear_cuda_memory(args.device)
        revision = revision_candidates[0] if mxq_path else _select_revision(model_id, revision_candidates)
        if args.all and not args.mxq_dir and revision is None:
            print(f"Skipping {label} (missing revisions).")
            continue
        json_path = Path(results_dir) / f"{base}_measure.json"
        if args.skip_existing and json_path.is_file():
            print(f"Skipping {label} (measure result exists).")
            continue
        if _should_precheck_cuda(args):
            estimated = _estimate_model_weight_bytes(model_id, revision)
            mem_info = _cuda_memory_info(args.device)
            if estimated is not None and mem_info is not None:
                free_b, _ = mem_info
                required = int(float(estimated) * float(args.cuda_precheck_margin))
                if free_b < required:
                    print(
                        f"Skipping {label} (pre-check VRAM insufficient): "
                        f"free={_format_gib(free_b)} required~={_format_gib(required)}"
                    )
                    _clear_cuda_memory(args.device)
                    continue
        pipeline = None
        try:
            pipeline = _build_pipeline(
                model_id,
                tokenizer=args.tokenizer,
                revision=revision,
                device=args.device,
                device_map=args.device_map,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                core_mode=core_mode,
                mxq_path=mxq_path,
            )
            measurer = TPSMeasurer(pipeline)
            resolved_prefill_chunk_size = None if disable_npu_specific_args else args.prefill_chunk_size
            for i in tqdm(range(args.warmup), desc=f"{label} warmup", leave=False):
                measurer.measure(
                    num_prefill=args.prefill,
                    num_decode=args.decode,
                    batch_size=batch_size,
                    prefill_chunk_size=resolved_prefill_chunk_size,
                    show_progress=True,
                    progress_desc=f"{label} warmup generate {i + 1}/{args.warmup}",
                )
            runs: list[dict[str, Any]] = []
            device_time_series_runs: list[dict[str, Any]] = []
            for repeat_idx in tqdm(range(args.repeat), desc=f"{label} measured runs", leave=False):
                tracker_prefill, tracker_decode = _build_phase_trackers(target_args, pipeline)
                try:
                    run = measurer.measure(
                        num_prefill=args.prefill,
                        num_decode=args.decode,
                        batch_size=batch_size,
                        prefill_chunk_size=resolved_prefill_chunk_size,
                        show_progress=True,
                        progress_desc=f"{label} run {repeat_idx + 1}/{args.repeat}",
                        on_prefill_start=(lambda: tracker_prefill.start()) if tracker_prefill is not None else None,
                        on_prefill_end=(lambda: tracker_prefill.stop()) if tracker_prefill is not None else None,
                        on_decode_start=(lambda: tracker_decode.start()) if tracker_decode is not None else None,
                        on_decode_end=(lambda: tracker_decode.stop()) if tracker_decode is not None else None,
                    )
                finally:
                    _stop_tracker_safe(tracker_prefill)
                    _stop_tracker_safe(tracker_decode)
                row = asdict(run)
                if tracker_prefill is not None and tracker_decode is not None:
                    prefill_metric = _extract_device_metric(tracker_prefill)
                    decode_metric = _extract_device_metric(tracker_decode)
                    row["avg_power_w"] = _weighted_two(
                        prefill_metric.get("avg_power_w"),
                        run.prefill_latency,
                        decode_metric.get("avg_power_w"),
                        run.decode_duration,
                    )
                    row["p99_power_w"] = max(
                        [
                            v
                            for v in (prefill_metric.get("p99_power_w"), decode_metric.get("p99_power_w"))
                            if v is not None
                        ],
                        default=None,
                    )
                    row["avg_utilization_pct"] = _weighted_two(
                        prefill_metric.get("avg_utilization_pct"),
                        run.prefill_latency,
                        decode_metric.get("avg_utilization_pct"),
                        run.decode_duration,
                    )
                    row["avg_memory_used_mb"] = _weighted_two(
                        prefill_metric.get("avg_memory_used_mb"),
                        run.prefill_latency,
                        decode_metric.get("avg_memory_used_mb"),
                        run.decode_duration,
                    )
                    row["total_energy_j"] = (
                        (float(row["avg_power_w"]) * run.total_time)
                        if isinstance(row.get("avg_power_w"), (int, float))
                        else None
                    )
                    row["prefill_tokens_per_j"] = (
                        _safe_div(run.prefill_tps, float(prefill_metric["avg_power_w"]))
                        if isinstance(prefill_metric.get("avg_power_w"), (int, float))
                        else None
                    )
                    row["decode_tokens_per_j"] = (
                        _safe_div(run.decode_tps, float(decode_metric["avg_power_w"]))
                        if isinstance(decode_metric.get("avg_power_w"), (int, float))
                        else None
                    )
                    row["prefill_j_per_token"] = (
                        _safe_div(1.0, float(row["prefill_tokens_per_j"]))
                        if isinstance(row.get("prefill_tokens_per_j"), (int, float))
                        else None
                    )
                    row["decode_j_per_token"] = (
                        _safe_div(1.0, float(row["decode_tokens_per_j"]))
                        if isinstance(row.get("decode_tokens_per_j"), (int, float))
                        else None
                    )
                    device_time_series_runs.append(
                        {
                            "prefill": _extract_device_time_series(tracker_prefill),
                            "decode": _extract_device_time_series(tracker_decode),
                        }
                    )
                runs.append(row)
            payload = {
                "model": label,
                "benchmark_type": "measure",
                "task": "text-generation",
                "batch_mode": args.batch_mode,
                "batch_size": batch_size,
                "prefill": args.prefill,
                "decode": args.decode,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "runs": runs,
                "summary": {
                    "prefill_tps": _summary([r["prefill_tps"] for r in runs]),
                    "decode_tps": _summary([r["decode_tps"] for r in runs]),
                    "ttft_ms": _summary([r["prefill_latency"] * 1000.0 for r in runs]),
                    "decode_duration_ms": _summary([r["decode_duration"] * 1000.0 for r in runs]),
                    "total_time_ms": _summary([r["total_time"] * 1000.0 for r in runs]),
                    "prefill_npu_latency_pct": _summary(
                        [r["prefill_npu_latency_pct"] for r in runs if r.get("prefill_npu_latency_pct") is not None]
                    ),
                    "decode_npu_latency_pct": _summary(
                        [r["decode_npu_latency_pct"] for r in runs if r.get("decode_npu_latency_pct") is not None]
                    ),
                },
                "device": _measure_device_payload(runs),
                "device_time_series_runs": device_time_series_runs,
            }
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Saved: {json_path.name}")
        except Exception as e:
            print(f"Skipping {label} (measure failed): {e}")
        finally:
            _release_pipeline(pipeline, args.device)
    _rebuild_measure_outputs(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
