import argparse
import csv
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

from transformers import pipeline as hf_pipeline
from tqdm import tqdm

from chart_utils import collect_folder_metrics, plot_scalar_chart, plot_token_chart
from mblt_model_zoo.hf_transformers.utils import list_models
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    add_device_tracking_args as _add_device_tracking_args,
    add_pipeline_device_args as _add_pipeline_device_args,
    build_device_tracker as _build_device_tracker_common,
    build_phase_trackers as _build_phase_trackers_common,
    extract_device_metric as _extract_device_metric_common,
    parse_positive_int as _parse_positive_int_common,
    parse_positive_int_optional as _parse_positive_int_optional_common,
    print_device_status as _print_device_status_common,
    stop_tracker_safe as _stop_tracker_safe_common,
    weighted_two as _weighted_two_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
    BenchmarkResult,
    SweepData,
    TPSMeasurer,
)

DEFAULT_FALLBACK_CHUNK_SIZE = 128


def _safe_filename(model_id: str) -> str:
    return model_id.replace("/", "__")


def _parse_range_arg(raw: str) -> Tuple[int, int, int]:
    sep = ":" if ":" in raw else ("," if "," in raw else None)
    if sep is None:
        raise argparse.ArgumentTypeError(
            "expected format 'start:end:step' or 'start,end,step'"
        )
    parts = [p.strip() for p in raw.split(sep)]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected exactly 3 integers: 'start:end:step' or 'start,end,step'"
        )
    try:
        start, end, step = (int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("range values must be integers") from e
    if start <= 0 or end <= 0 or step <= 0:
        raise argparse.ArgumentTypeError("range values must be positive integers")
    if start > end:
        raise argparse.ArgumentTypeError("range start must be <= end")
    return start, end, step


def _parse_positive_int(raw: str) -> int:
    return _parse_positive_int_common(raw)


def _parse_positive_int_optional(raw: str) -> int | None:
    return _parse_positive_int_optional_common(raw)


def _build_pipeline(
    model_id: str,
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
    if device_map:
        kwargs["device_map"] = device_map
    model_kwargs: dict[str, Any] = {}
    if core_mode:
        model_kwargs["core_mode"] = core_mode
        if core_mode == "single":
            model_kwargs["target_cores"] = ["0:0"]
        elif core_mode == "global4":
            model_kwargs["target_clusters"] = [0]
        elif core_mode == "global8":
            model_kwargs["target_clusters"] = [0, 1]
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


def _is_cuda_device(device: str | None) -> bool:
    return isinstance(device, str) and device.strip().lower().startswith("cuda")


def _cuda_device_index(device: str | None) -> int | None:
    if not _is_cuda_device(device):
        return None
    text = (device or "").strip().lower()
    if ":" not in text:
        return 0
    try:
        return int(text.split(":", 1)[1])
    except ValueError:
        return None


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or ("out of memory" in msg and "cuda" in msg)
        or "cublas_status_alloc_failed" in msg
    )


def _clear_cuda_memory(device: str | None) -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    idx = _cuda_device_index(device)
    try:
        if idx is not None:
            torch.cuda.set_device(idx)
    except Exception:
        pass
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
    return f"{float(num_bytes) / (1024 ** 3):.2f} GiB"


def _should_precheck_cuda(args: argparse.Namespace) -> bool:
    if not args.cuda_precheck:
        return False
    if _is_cuda_device(args.device):
        return True
    # If only device_map is set (e.g. auto), target GPU topology is ambiguous.
    return False


def _load_chunk_size_lookup_csv(path: str | None) -> dict[tuple[str, str, str], int]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.is_file():
        print(f"Warning: chunk-size lookup CSV not found: {csv_path}")
        return {}
    out: dict[tuple[str, str, str], int] = {}
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = str(row.get("model_id", "")).strip()
                revision = str(row.get("revision", "")).strip()
                core_mode = str(row.get("core_mode", "")).strip()
                raw_chunk = row.get("best_chunk_size")
                if not model or not revision or not core_mode:
                    continue
                try:
                    chunk = int(float(str(raw_chunk)))
                except Exception:
                    continue
                if chunk <= 0:
                    continue
                out[(model, revision, core_mode)] = chunk
    except Exception as e:
        print(f"Warning: failed to load chunk-size lookup CSV: {e}")
        return {}
    print(f"Loaded chunk-size lookup entries: {len(out)} from {csv_path}")
    return out


def _resolve_lookup_chunk_size(
    lookup: dict[tuple[str, str, str], int],
    *,
    model_id: str,
    revision: str | None,
    core_mode: str | None,
) -> int | None:
    if not lookup:
        return None
    if core_mode is None:
        return None
    rev = "" if revision is None else str(revision)
    return lookup.get((model_id, rev, core_mode))


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
    elif hasattr(card_data, "to_dict"):
        try:
            payload = card_data.to_dict()
        except Exception:
            payload = None
    elif hasattr(card_data, "__dict__"):
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

    revision_map = [
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
    basename_matches = [
        m for m in available_model_ids if m.split("/", 1)[-1] == model_part
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]
    basename_matches_slash = [
        m for m in available_model_ids if m.split("/", 1)[-1] == model_part_slash
    ]
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
                f"Skipping mxq (cannot resolve model_id from filename): {path.name} "
                f"(expected <model_id>-<W8|W4V8>.mxq)"
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


def _safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def _extract_device_metric(tracker: Any) -> dict[str, float | None]:
    return _extract_device_metric_common(tracker)


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


def _write_device_combined_markdown(path: str, rows: Sequence[dict[str, float | str | None]]) -> None:
    if not rows:
        return
    lines = [
        "| model | avg_power_w | p99_power_w | avg_utilization_pct | p99_utilization_pct | avg_memory_used_mb | p99_memory_used_mb | total_memory_mb | avg_memory_used_pct | p99_memory_used_pct | total_energy_j | prefill_tps_last | decode_tps_last | prefill_tok_per_j_last | decode_tok_per_j_last | prefill_j_per_tok_last | decode_j_per_tok_last |\n",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in rows:
        lines.append(
            "| {model} | {avg_power_w} | {p99_power_w} | {avg_utilization_pct} | {p99_utilization_pct} | {avg_memory_used_mb} | {p99_memory_used_mb} | {total_memory_mb} | {avg_memory_used_pct} | {p99_memory_used_pct} | {total_energy_j} | {prefill_tps_last} | {decode_tps_last} | {prefill_tok_per_j_last} | {decode_tok_per_j_last} | {prefill_j_per_tok_last} | {decode_j_per_tok_last} |\n".format(
                model=row["model"],
                avg_power_w="" if row["avg_power_w"] is None else f"{row['avg_power_w']:.6f}",
                p99_power_w="" if row["p99_power_w"] is None else f"{row['p99_power_w']:.6f}",
                avg_utilization_pct=""
                if row["avg_utilization_pct"] is None
                else f"{row['avg_utilization_pct']:.6f}",
                p99_utilization_pct=""
                if row["p99_utilization_pct"] is None
                else f"{row['p99_utilization_pct']:.6f}",
                avg_memory_used_mb=""
                if row["avg_memory_used_mb"] is None
                else f"{row['avg_memory_used_mb']:.6f}",
                p99_memory_used_mb=""
                if row["p99_memory_used_mb"] is None
                else f"{row['p99_memory_used_mb']:.6f}",
                total_memory_mb=""
                if row["total_memory_mb"] is None
                else f"{row['total_memory_mb']:.6f}",
                avg_memory_used_pct=""
                if row["avg_memory_used_pct"] is None
                else f"{row['avg_memory_used_pct']:.6f}",
                p99_memory_used_pct=""
                if row["p99_memory_used_pct"] is None
                else f"{row['p99_memory_used_pct']:.6f}",
                total_energy_j="" if row["total_energy_j"] is None else f"{row['total_energy_j']:.6f}",
                prefill_tps_last="" if row["prefill_tps_last"] is None else f"{row['prefill_tps_last']:.6f}",
                decode_tps_last="" if row["decode_tps_last"] is None else f"{row['decode_tps_last']:.6f}",
                prefill_tok_per_j_last="" if row["prefill_tok_per_j_last"] is None else f"{row['prefill_tok_per_j_last']:.6f}",
                decode_tok_per_j_last="" if row["decode_tok_per_j_last"] is None else f"{row['decode_tok_per_j_last']:.6f}",
                prefill_j_per_tok_last="" if row["prefill_j_per_tok_last"] is None else f"{row['prefill_j_per_tok_last']:.6f}",
                decode_j_per_tok_last="" if row["decode_j_per_tok_last"] is None else f"{row['decode_j_per_tok_last']:.6f}",
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_single_combined_markdown(
    path: str,
    tps_rows: Sequence[dict[str, Any]],
    device_rows: Sequence[dict[str, float | str | None]],
) -> None:
    if not tps_rows:
        return
    models = sorted({str(r["model"]) for r in tps_rows})
    prefill_tokens = sorted(
        {
            int(r["tokens"])
            for r in tps_rows
            if str(r.get("phase")) == "prefill" and isinstance(r.get("tokens"), int)
        }
    )
    decode_tokens = sorted(
        {
            int(r["tokens"])
            for r in tps_rows
            if str(r.get("phase")) == "decode" and isinstance(r.get("tokens"), int)
        }
    )
    tps_map: dict[tuple[str, str, int], float] = {}
    time_map: dict[tuple[str, str, int], float] = {}
    for row in tps_rows:
        model = str(row["model"])
        phase = str(row["phase"])
        token = int(row["tokens"])
        tps_val = row.get("tps")
        time_ms_val = row.get("time_ms")
        if isinstance(tps_val, (int, float)):
            tps_map[(model, phase, token)] = float(tps_val)
        if isinstance(time_ms_val, (int, float)):
            time_map[(model, phase, token)] = float(time_ms_val)

    device_map = {
        str(r["model"]): r for r in device_rows if isinstance(r.get("model"), str)
    }
    device_cols = [
        "avg_power_w",
        "p99_power_w",
        "avg_utilization_pct",
        "p99_utilization_pct",
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

        drow = device_map.get(model, {})
        for col in device_cols:
            v = drow.get(col) if isinstance(drow, dict) else None
            values.append("" if not isinstance(v, (int, float)) else f"{float(v):.6f}")
        lines.append("| " + " | ".join(values) + " |\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _rebuild_combined_outputs(
    results_dir: str,
    targets: Sequence[tuple[str, list[str | None], str, str, str | None]],
) -> None:
    combined_results = []
    combined_rows = []
    combined_device_rows: list[dict[str, float | str | None]] = []
    for _, _, label, base, _ in targets:
        json_path = os.path.join(results_dir, f"{base}.json")
        if not os.path.isfile(json_path):
            continue
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
        return

    combined_csv = os.path.join(results_dir, "combined.csv")
    combined_md = os.path.join(results_dir, "combined.md")
    BenchmarkResult.write_combined_csv(combined_csv, combined_rows)
    _write_single_combined_markdown(
        combined_md,
        tps_rows=combined_rows,
        device_rows=combined_device_rows,
    )

    folder_metrics = collect_folder_metrics(Path(results_dir))
    if folder_metrics:
        models = sorted(folder_metrics.keys())
        labels = ["benchmark"]
        metrics_by_folder = [folder_metrics]

        plot_token_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            token_selector=lambda m: m.prefill_tps,
            title="Prefill TPS",
            x_label="TPS (tokens/sec)",
            output_path=Path(results_dir) / "prefill_tps.png",
        )
        plot_token_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            token_selector=lambda m: m.decode_tps,
            title="Decode TPS",
            x_label="TPS (tokens/sec)",
            output_path=Path(results_dir) / "decode_tps.png",
        )
        plot_token_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            token_selector=lambda m: m.prefill_latency_ms,
            title="Prefill Latency",
            x_label="Latency (ms)",
            output_path=Path(results_dir) / "prefill_latency_ms.png",
        )
        plot_token_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            token_selector=lambda m: m.decode_duration_ms,
            title="Decode Duration",
            x_label="Duration (ms)",
            output_path=Path(results_dir) / "decode_duration_ms.png",
        )

        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.avg_power_w,
            title="Average Power",
            x_label="Power (W)",
            output_path=Path(results_dir) / "avg_power_w.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.total_energy_j,
            title="Total Energy",
            x_label="Energy (J)",
            output_path=Path(results_dir) / "total_energy_j.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.prefill_tokens_per_j,
            title="Prefill Tokens/J",
            x_label="Tokens/J",
            output_path=Path(results_dir) / "prefill_tokens_per_j.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.decode_tokens_per_j,
            title="Decode Tokens/J",
            x_label="Tokens/J",
            output_path=Path(results_dir) / "decode_tokens_per_j.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.prefill_j_per_token,
            title="Prefill J/Token",
            x_label="J/Token",
            output_path=Path(results_dir) / "prefill_j_per_token.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.decode_j_per_token,
            title="Decode J/Token",
            x_label="J/Token",
            output_path=Path(results_dir) / "decode_j_per_token.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.avg_utilization_pct,
            title="Average Utilization",
            x_label="Utilization (%)",
            output_path=Path(results_dir) / "avg_utilization_pct.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.p99_utilization_pct,
            title="P99 Utilization",
            x_label="Utilization (%)",
            output_path=Path(results_dir) / "p99_utilization_pct.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.avg_memory_used_mb,
            title="Average Memory Used",
            x_label="Memory (MB)",
            output_path=Path(results_dir) / "avg_memory_used_mb.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.p99_memory_used_mb,
            title="P99 Memory Used",
            x_label="Memory (MB)",
            output_path=Path(results_dir) / "p99_memory_used_mb.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.avg_memory_used_pct,
            title="Average Memory Used (%)",
            x_label="Memory Usage (%)",
            output_path=Path(results_dir) / "avg_memory_used_pct.png",
        )
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda m: m.p99_memory_used_pct,
            title="P99 Memory Used (%)",
            x_label="Memory Usage (%)",
            output_path=Path(results_dir) / "p99_memory_used_pct.png",
        )

    if combined_device_rows:
        device_csv = os.path.join(results_dir, "combined_device.csv")
        _write_device_combined_csv(device_csv, combined_device_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark text-generation models.")
    _add_pipeline_device_args(parser, device_default=None, trust_remote_code_default=True)
    parser.add_argument(
        "--revision",
        default=None,
        help="model revision (e.g., W8)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="benchmark W8 and W4V8 revisions only (skip main)",
    )
    parser.add_argument(
        "--mxq-dir",
        default=None,
        help=(
            "directory containing local mxq files. "
            "When set, only files matching <model_id>-<W8|W4V8>.mxq are benchmarked."
        ),
    )
    parser.add_argument(
        "--prefill-range",
        type=_parse_range_arg,
        default=(128, 512, 128),
        help="prefill sweep range as 'start:end:step' (default: 128:512:128)",
    )
    parser.add_argument(
        "--decode-range",
        type=_parse_range_arg,
        default=(128, 512, 128),
        help="decode sweep range as 'start:end:step' (default: 128:512:128)",
    )
    parser.add_argument(
        "--fixed-decode",
        type=_parse_positive_int,
        default=10,
        help="fixed decode length used during prefill sweep",
    )
    parser.add_argument(
        "--fixed-prefill",
        type=_parse_positive_int,
        default=128,
        help="fixed prefill length used during decode sweep",
    )
    parser.add_argument(
        "--chunk-size",
        type=_parse_positive_int_optional,
        default=None,
        help="optional chunk_size forwarded to model.generate/model.forward (default: None)",
    )
    parser.add_argument(
        "--core-mode",
        choices=["single", "global4", "global8"],
        default="global8",
        help="core mode passed to model_kwargs (default: global8)",
    )
    parser.add_argument(
        "--chunk-size-lookup-csv",
        default="prefill_chunk_size.csv",
        help=(
            "CSV path for per-model chunk_size lookup. "
            "Expected columns: model_id,revision,core_mode,best_chunk_size"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip models with existing JSON+PNG outputs",
    )
    parser.add_argument(
        "--rebuild-charts",
        action="store_true",
        help="skip benchmarking and rebuild combined CSV/MD/charts from existing JSON files",
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
        help=(
            "resolve each Mobilint model to its parent/base model from HF Hub and "
            "benchmark the unique parent model ids"
        ),
    )
    _add_device_tracking_args(parser)
    parser.add_argument(
        "--cuda-precheck",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "best-effort CUDA VRAM pre-check before loading each model "
            "(default: on, disable via --no-cuda-precheck)"
        ),
    )
    parser.add_argument(
        "--cuda-precheck-margin",
        type=float,
        default=1.15,
        help="required free VRAM factor versus estimated model weights (default: 1.15)",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    lookup_path = args.chunk_size_lookup_csv
    if lookup_path and not os.path.isabs(lookup_path):
        lookup_path = os.path.join(os.path.dirname(__file__), lookup_path)
    chunk_lookup = _load_chunk_size_lookup_csv(lookup_path)

    available = list_models(tasks="text-generation")
    available_model_ids = available.get("text-generation", [])
    if not available_model_ids:
        print("No text-generation models found.")
        return 0

    results_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        "text_generation",
    )
    os.makedirs(results_dir, exist_ok=True)

    if args.mxq_dir:
        mxq_dir = Path(args.mxq_dir).expanduser().resolve()
        if not mxq_dir.is_dir():
            raise SystemExit(f"--mxq-dir is not a directory: {mxq_dir}")
        if args.original_models or args.all or args.revision:
            print(
                "Note: --mxq-dir is set, so --original-models/--all/--revision are ignored "
                "(revision is taken from mxq filename suffix)."
            )
        targets = _iter_targets_from_mxq_dir(
            mxq_dir=mxq_dir,
            available_model_ids=available_model_ids,
        )
        if not targets:
            raise SystemExit(
                "No valid mxq targets found. Expected files named "
                "<model_id>-<W8|W4V8>.mxq in --mxq-dir."
            )
        print(f"Using local mxq targets from {mxq_dir}: {len(targets)} files")
    else:
        model_ids = available_model_ids
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
        targets = list(
            _iter_targets(
                model_ids,
                revision=args.revision,
                all_revisions=args.all,
            )
        )
    if args.rebuild_charts:
        print("Rebuilding combined outputs from existing JSON files only...")
        _rebuild_combined_outputs(results_dir, targets)
        return 0

    for model_id, revision_candidates, label, base, mxq_path in tqdm(
        targets,
        desc="Benchmarking models",
        total=len(targets),
        unit="model",
    ):
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
        if (
            args.skip_existing
            and os.path.isfile(json_path)
            and os.path.isfile(png_path)
        ):
            print("Skipping (results exist).")
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
            try:
                pipeline = _build_pipeline(
                    model_id,
                    revision=revision,
                    device=args.device,
                    device_map=args.device_map,
                    dtype=args.dtype,
                    trust_remote_code=args.trust_remote_code,
                    core_mode=args.core_mode,
                    mxq_path=mxq_path,
                )
            except Exception as e:
                if _is_cuda_oom_error(e):
                    print(f"Skipping (CUDA OOM while loading model): {e}")
                    _clear_cuda_memory(args.device)
                    continue
                if args.all and not args.mxq_dir and _revision_exists(model_id, revision or "") is None:
                    print(f"Skipping (failed to load revision {revision}): {e}")
                else:
                    print(f"Skipping (failed to load model): {e}")
                continue

            measurer = TPSMeasurer(pipeline)
            tracker_prefill, tracker_decode = _build_phase_trackers(args, pipeline)
            _print_device_status(args, tracker_prefill)
            lookup_chunk = _resolve_lookup_chunk_size(
                chunk_lookup,
                model_id=model_id,
                revision=revision,
                core_mode=args.core_mode,
            )
            resolved_chunk_size = args.chunk_size
            if resolved_chunk_size is None and lookup_chunk is not None:
                resolved_chunk_size = lookup_chunk
                print(
                    f"Using chunk-size lookup: model={model_id} revision={revision} "
                    f"core_mode={args.core_mode} chunk_size={resolved_chunk_size}"
                )
            elif args.chunk_size is None and args.chunk_size_lookup_csv and lookup_chunk is None:
                resolved_chunk_size = DEFAULT_FALLBACK_CHUNK_SIZE
                print(
                    f"Warning: no chunk-size lookup match for model={model_id} "
                    f"revision={revision} core_mode={args.core_mode}; "
                    f"using fallback chunk_size={resolved_chunk_size}"
                )
            elif args.chunk_size is not None and lookup_chunk is not None:
                print(
                    f"Note: --chunk-size={args.chunk_size} overrides lookup chunk_size={lookup_chunk}"
                )
            for i in tqdm(range(args.warmup), desc=f"{label} warmup", leave=False):
                measurer.measure(
                    num_prefill=args.fixed_prefill,
                    num_decode=args.fixed_decode,
                    chunk_size=resolved_chunk_size,
                    trace_path=None,
                    show_progress=True,
                    progress_desc=f"{label} warmup generate {i + 1}/{args.warmup}",
                )
            try:
                result = measurer.measure_full(
                    prefill_range=args.prefill_range,
                    decode_range=args.decode_range,
                    fixed_decode_len=args.fixed_decode,
                    fixed_prefill_len=args.fixed_prefill,
                    chunk_size=resolved_chunk_size,
                    show_progress=True,
                    progress_prefix=label,
                    on_prefill_start=(lambda: tracker_prefill.start()) if tracker_prefill is not None else None,
                    on_prefill_end=(lambda: tracker_prefill.stop()) if tracker_prefill is not None else None,
                    on_decode_start=(lambda: tracker_decode.start()) if tracker_decode is not None else None,
                    on_decode_end=(lambda: tracker_decode.stop()) if tracker_decode is not None else None,
                )
            finally:
                _stop_tracker_safe(tracker_prefill)
                _stop_tracker_safe(tracker_decode)
        except Exception as e:
            if _is_cuda_oom_error(e):
                print(f"Skipping (CUDA OOM during benchmark): {e}")
                _release_pipeline(pipeline, args.device)
                continue
            print(f"Skipping (benchmark failed): {e}")
            _release_pipeline(pipeline, args.device)
            continue

        if result.prefill_sweep.avg_total_token_latency_values:
            avg_total = result.prefill_sweep.avg_total_token_latency_values[-1]
            avg_npu = result.prefill_sweep.avg_npu_token_latency_values[-1]
            avg_npu_str = f"{avg_npu * 1000.0:.3f}ms" if avg_npu is not None else "n/a"
            print(
                "Avg prefill token latency (last): "
                f"total={avg_total * 1000.0:.3f}ms npu={avg_npu_str}"
            )
        if result.decode_sweep.avg_total_token_latency_values:
            avg_total = result.decode_sweep.avg_total_token_latency_values[-1]
            avg_npu = result.decode_sweep.avg_npu_token_latency_values[-1]
            avg_npu_str = f"{avg_npu * 1000.0:.3f}ms" if avg_npu is not None else "n/a"
            print(
                "Avg decode token latency (last): "
                f"total={avg_total * 1000.0:.3f}ms npu={avg_npu_str}"
            )
        device_payload: dict[str, float | None] | None = None
        if tracker_prefill is not None and tracker_decode is not None:
            prefill_metric = _extract_device_metric(tracker_prefill)
            decode_metric = _extract_device_metric(tracker_decode)
            prefill_phase_duration_s = float(getattr(result, "prefill_phase_duration_s", 0.0) or 0.0)
            decode_phase_duration_s = float(getattr(result, "decode_phase_duration_s", 0.0) or 0.0)
            prefill_avg_power = prefill_metric.get("avg_power_w")
            decode_avg_power = decode_metric.get("avg_power_w")
            prefill_energy = (
                prefill_avg_power * prefill_phase_duration_s
                if prefill_avg_power is not None
                else None
            )
            decode_energy = (
                decode_avg_power * decode_phase_duration_s
                if decode_avg_power is not None
                else None
            )
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
                [v for v in (prefill_metric.get("total_memory_mb"), decode_metric.get("total_memory_mb")) if v is not None],
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
            prefill_last = (
                float(result.prefill_sweep.tps_values[-1])
                if result.prefill_sweep.tps_values
                else None
            )
            decode_last = (
                float(result.decode_sweep.tps_values[-1])
                if result.decode_sweep.tps_values
                else None
            )
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
                "prefill_avg_memory_used_mb": prefill_metric.get("avg_memory_used_mb"),
                "prefill_p99_memory_used_mb": prefill_metric.get("p99_memory_used_mb"),
                "prefill_avg_memory_used_pct": prefill_metric.get("avg_memory_used_pct"),
                "prefill_p99_memory_used_pct": prefill_metric.get("p99_memory_used_pct"),
                "decode_avg_power_w": decode_metric.get("avg_power_w"),
                "decode_p99_power_w": decode_metric.get("p99_power_w"),
                "decode_avg_utilization_pct": decode_metric.get("avg_utilization_pct"),
                "decode_p99_utilization_pct": decode_metric.get("p99_utilization_pct"),
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
            "benchmark": asdict(result),
            "device": device_payload,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        measurer.plot_and_save(result, save_path=png_path)

        _release_pipeline(pipeline, args.device)

    _rebuild_combined_outputs(results_dir, targets)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
