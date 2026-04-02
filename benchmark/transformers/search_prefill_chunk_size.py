from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

from mblt_model_zoo.hf_transformers.utils import list_models
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    add_pipeline_device_args as _add_pipeline_device_args,
    apply_core_mode_model_kwargs as _apply_core_mode_model_kwargs_common,
    CORE_MODE_SWEEP_VALUES as _CORE_MODE_SWEEP_VALUES_COMMON,
    parse_positive_int as _parse_positive_int_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import TPSMeasurer


def _safe_filename(text: str) -> str:
    return text.replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")


def _discover_targets_from_mxq_dir(
    *,
    mxq_dir: Path,
    available_model_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    basename_to_model_ids: dict[str, list[str]] = {}
    for model_id in available_model_ids:
        base = str(model_id).split("/")[-1]
        basename_to_model_ids.setdefault(base, []).append(str(model_id))

    targets: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    pattern = re.compile(r"^(?P<base>.+)-(?P<rev>W8|W4V8)\.mxq$")
    mxq_files = sorted(mxq_dir.glob("*.mxq"))
    for mxq_path in mxq_files:
        filename = mxq_path.name
        matched = pattern.match(filename)
        if not matched:
            skipped.append(
                {
                    "mxq_file": filename,
                    "reason": "invalid filename format (expected: <model_without_group>-<W8|W4V8>.mxq)",
                }
            )
            continue
        model_base = str(matched.group("base"))
        revision = str(matched.group("rev"))
        model_matches = basename_to_model_ids.get(model_base, [])
        if not model_matches:
            skipped.append(
                {
                    "mxq_file": filename,
                    "reason": f"model base '{model_base}' not found in text-generation HF model list",
                }
            )
            continue
        if len(model_matches) > 1:
            skipped.append(
                {
                    "mxq_file": filename,
                    "reason": f"ambiguous model base '{model_base}' maps to multiple model ids: {model_matches}",
                }
            )
            continue
        targets.append(
            {
                "mxq_file": filename,
                "mxq_path": str(mxq_path.resolve()),
                "model_id": model_matches[0],
                "revision": revision,
            }
        )
    return targets, skipped


def _parse_positive_int(raw: str) -> int:
    return _parse_positive_int_common(raw)


def _parse_core_modes(raw: str) -> list[str]:
    modes = [x.strip() for x in raw.split(",") if x.strip()]
    if not modes:
        raise argparse.ArgumentTypeError("at least one core mode is required")
    allowed = set(_CORE_MODE_SWEEP_VALUES_COMMON)
    invalid = [m for m in modes if m not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported core mode(s): {invalid}. allowed={sorted(allowed)}")
    return modes


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


def _build_pipeline(
    *,
    model_id: str,
    revision: str | None,
    core_mode: str,
    device: str | None,
    device_map: str | None,
    dtype: str | None,
    trust_remote_code: bool,
    mxq_path: str | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "task": "text-generation",
        "model": model_id,
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    if device is not None:
        kwargs["device"] = device
    if device_map:
        kwargs["device_map"] = device_map

    model_kwargs: dict[str, Any] = {}
    model_kwargs = _apply_core_mode_model_kwargs_common(model_kwargs, core_mode)
    if mxq_path:
        model_kwargs["mxq_path"] = mxq_path
    kwargs["model_kwargs"] = model_kwargs

    if dtype:
        try:
            kwargs["dtype"] = dtype
            return hf_pipeline(**kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = dtype
            return hf_pipeline(**kwargs)
    return hf_pipeline(**kwargs)


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


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _measure_prefill_once(
    *,
    measurer: TPSMeasurer,
    prefill_length: int,
    decode_length: int,
    chunk_size: int,
) -> tuple[float, float]:
    t0 = time.perf_counter()
    single = measurer.measure(
        num_prefill=prefill_length,
        num_decode=decode_length,
        chunk_size=chunk_size,
        show_progress=False,
    )
    t1 = time.perf_counter()
    return float(single.prefill_tps), float(t1 - t0)


def _measure_prefill_median(
    *,
    measurer: TPSMeasurer,
    prefill_length: int,
    decode_length: int,
    chunk_size: int,
    repeat: int,
) -> tuple[float, list[float], list[float]]:
    tps_values: list[float] = []
    wall_times: list[float] = []
    for _ in range(repeat):
        tps, wall = _measure_prefill_once(
            measurer=measurer,
            prefill_length=prefill_length,
            decode_length=decode_length,
            chunk_size=chunk_size,
        )
        tps_values.append(tps)
        wall_times.append(wall)
    return _median(tps_values), tps_values, wall_times


def _is_timeout_like_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = (
        "model_notalive",
        "timeout",
        "npu_watchdog",
        "fatal error",
        "qbruntimeerror",
        "device hang",
    )
    return any(k in msg for k in keywords)


def _parse_int_csv(raw: str) -> list[int]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        values = [int(x) for x in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError("all values must be integers") from e
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("all values must be > 0")
    return sorted(set(values))


def _run_prefill_grid(
    *,
    measurer: TPSMeasurer,
    prefill_lengths: list[int],
    chunk_candidates: list[int],
    decode_length: int,
    repeat: int,
    time_guard_sec: float,
    progress_bar: Any | None = None,
    log_prefix: str = "",
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    best_by_prefill: list[dict[str, Any]] = []
    global_failed_chunks: list[dict[str, Any]] = []

    def _touch_progress(prefill_length: int, chunk: int, label: str, tps: float | None) -> None:
        if progress_bar is None:
            return
        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                "prefill": prefill_length,
                "chunk": chunk,
                "stage": label,
                "score": "fail" if tps is None else f"{tps:.3f}",
                "evals": len(evaluations),
            }
        )

    for prefill_length in prefill_lengths:
        print(f"[search] {log_prefix} prefill={prefill_length} chunks={chunk_candidates} guard={time_guard_sec:.1f}s")
        cutoff_chunk: int | None = None
        for chunk_size in chunk_candidates:
            if int(chunk_size) > int(prefill_length):
                row = {
                    "prefill_length": int(prefill_length),
                    "chunk_size": int(chunk_size),
                    "median_tps": float("-inf"),
                    "tps_values": [],
                    "wall_time_values_s": [],
                    "failed": True,
                    "too_slow": False,
                    "error": "skipped because chunk_size > prefill_length",
                    "stage": "grid-skip",
                }
                evaluations.append(row)
                global_failed_chunks.append(
                    {
                        "prefill_length": int(prefill_length),
                        "chunk_size": int(chunk_size),
                        "reason": "chunk-greater-than-prefill",
                    }
                )
                _touch_progress(prefill_length, chunk_size, "skip", None)
                continue
            if cutoff_chunk is not None and chunk_size > cutoff_chunk:
                row = {
                    "prefill_length": int(prefill_length),
                    "chunk_size": int(chunk_size),
                    "median_tps": float("-inf"),
                    "tps_values": [],
                    "wall_time_values_s": [],
                    "failed": True,
                    "too_slow": True,
                    "error": f"skipped because smaller chunk exceeded {time_guard_sec:.1f}s",
                    "stage": "grid-skip",
                }
                evaluations.append(row)
                global_failed_chunks.append(
                    {"prefill_length": int(prefill_length), "chunk_size": int(chunk_size), "reason": "time-guard-skip"}
                )
                _touch_progress(prefill_length, chunk_size, "skip", None)
                continue
            try:
                med_tps, raw_tps, raw_wall = _measure_prefill_median(
                    measurer=measurer,
                    prefill_length=int(prefill_length),
                    decode_length=int(decode_length),
                    chunk_size=int(chunk_size),
                    repeat=int(repeat),
                )
                too_slow = any(float(w) > float(time_guard_sec) for w in raw_wall)
                row = {
                    "prefill_length": int(prefill_length),
                    "chunk_size": int(chunk_size),
                    "median_tps": float(med_tps),
                    "tps_values": [float(x) for x in raw_tps],
                    "wall_time_values_s": [float(x) for x in raw_wall],
                    "failed": False,
                    "too_slow": bool(too_slow),
                    "error": None,
                    "stage": "grid",
                }
                evaluations.append(row)
                _touch_progress(prefill_length, chunk_size, "grid", float(med_tps))
                if too_slow and cutoff_chunk is None:
                    cutoff_chunk = int(chunk_size)
                    print(
                        f"[search] {log_prefix} prefill={prefill_length} "
                        f"guard hit at chunk={chunk_size}. skip larger chunks."
                    )
            except Exception as e:
                is_timeout = _is_timeout_like_error(e)
                row = {
                    "prefill_length": int(prefill_length),
                    "chunk_size": int(chunk_size),
                    "median_tps": float("-inf"),
                    "tps_values": [],
                    "wall_time_values_s": [],
                    "failed": True,
                    "too_slow": bool(is_timeout),
                    "error": str(e),
                    "stage": "grid",
                }
                evaluations.append(row)
                global_failed_chunks.append(
                    {
                        "prefill_length": int(prefill_length),
                        "chunk_size": int(chunk_size),
                        "reason": str(e),
                    }
                )
                _touch_progress(prefill_length, chunk_size, "grid", None)
                if is_timeout and cutoff_chunk is None:
                    cutoff_chunk = int(chunk_size)
                    print(
                        f"[search] {log_prefix} prefill={prefill_length} "
                        f"timeout at chunk={chunk_size}. skip larger chunks."
                    )

        feasible = [
            x
            for x in evaluations
            if int(x["prefill_length"]) == int(prefill_length)
            and (not bool(x.get("failed")))
            and (not bool(x.get("too_slow")))
            and math.isfinite(float(x.get("median_tps", float("-inf"))))
        ]
        if feasible:
            best = max(feasible, key=lambda x: float(x["median_tps"]))
            best_by_prefill.append(
                {
                    "prefill_length": int(prefill_length),
                    "best_chunk_size": int(best["chunk_size"]),
                    "best_median_prefill_tps": float(best["median_tps"]),
                }
            )
            print(
                f"[search] {log_prefix} prefill={prefill_length} best_chunk={best['chunk_size']} "
                f"best_median_prefill_tps={float(best['median_tps']):.4f}"
            )
        else:
            best_by_prefill.append(
                {
                    "prefill_length": int(prefill_length),
                    "best_chunk_size": None,
                    "best_median_prefill_tps": None,
                }
            )
            print(f"[search] {log_prefix} prefill={prefill_length} no feasible chunk.")

    all_feasible_best = [x for x in best_by_prefill if x["best_chunk_size"] is not None]
    if not all_feasible_best:
        raise RuntimeError("No feasible chunk for any prefill length.")
    representative = max(all_feasible_best, key=lambda x: float(x["best_median_prefill_tps"]))
    return {
        "best_chunk_size": int(representative["best_chunk_size"]),
        "best_median_prefill_tps": float(representative["best_median_prefill_tps"]),
        "best_by_prefill": best_by_prefill,
        "evaluations": evaluations,
        "failed_chunks": global_failed_chunks,
        "time_guard_sec": float(time_guard_sec),
        "prefill_lengths": [int(x) for x in prefill_lengths],
        "chunk_candidates": [int(x) for x in chunk_candidates],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_record(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _collect_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    measurement_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("status")) != "ok":
            continue
        model_id = str(record["model_id"])
        revision = record.get("revision")
        core_mode = str(record["core_mode"])
        best_map = record.get("best_by_prefill", [])
        if isinstance(best_map, list):
            for b in best_map:
                best_rows.append(
                    {
                        "model_id": model_id,
                        "revision": "" if revision is None else revision,
                        "core_mode": core_mode,
                        "prefill_length": b.get("prefill_length"),
                        "best_chunk_size": b.get("best_chunk_size"),
                        "best_median_prefill_tps": b.get("best_median_prefill_tps"),
                    }
                )
        for ev in record.get("evaluations", []):
            chunk_size = int(ev["chunk_size"])
            prefill_length = int(ev.get("prefill_length", 0))
            measurement_rows.append(
                {
                    "model_id": model_id,
                    "revision": "" if revision is None else revision,
                    "core_mode": core_mode,
                    "prefill_length": prefill_length,
                    "chunk_size": chunk_size,
                    "median_prefill_tps": float(ev["median_tps"]),
                    "repeat_prefill_tps": ";".join(f"{float(x):.6f}" for x in ev.get("tps_values", [])),
                    "wall_time_values_s": ";".join(f"{float(x):.6f}" for x in ev.get("wall_time_values_s", [])),
                    "too_slow": int(bool(ev.get("too_slow"))),
                    "failed": int(bool(ev.get("failed"))),
                    "error": "" if ev.get("error") is None else str(ev.get("error")),
                }
            )
    return measurement_rows, best_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_mode_prefill_2d_chart(
    mode: str,
    prefill_length: int,
    measurement_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    revision: str | None = None,
) -> None:
    mode_rows = [
        r
        for r in measurement_rows
        if str(r.get("core_mode")) == mode
        and int(r.get("failed", 0)) == 0
        and int(r.get("prefill_length", -1)) == int(prefill_length)
        and (revision is None or str(r.get("revision", "") or "") == str(revision))
    ]
    if not mode_rows:
        return

    per_model: dict[str, list[tuple[int, float]]] = {}
    for row in mode_rows:
        model = str(row["model_id"])
        revision = str(row.get("revision", "") or "")
        key = f"{model}@{revision}" if revision else model
        x = int(row["chunk_size"])
        y = float(row["median_prefill_tps"])
        per_model.setdefault(key, []).append((x, y))

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("tab20")
    for idx, (model, points) in enumerate(sorted(per_model.items())):
        pts = sorted(points, key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.2,
            alpha=0.35,
            color=cmap(idx % 20),
            label=model,
        )

    rev_suffix = f", revision={revision}" if revision is not None else ""
    ax.set_title(f"Prefill TPS vs Chunk Size ({mode}, prefill={prefill_length}{rev_suffix})")
    ax.set_xlabel("chunk_size")
    ax.set_ylabel("median prefill TPS")
    ax.grid(True, linestyle="--", alpha=0.3)
    if per_model:
        if len(per_model) <= 12:
            ax.legend(loc="best", fontsize=8, title="model_id@revision")
        else:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=7,
                title="model_id@revision",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_best_chunk_chart(
    mode: str,
    best_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    revision: str | None = None,
) -> None:
    rows = []
    for r in best_rows:
        if str(r.get("core_mode")) != mode:
            continue
        rev = str(r.get("revision", "") or "")
        if revision is not None and rev != str(revision):
            continue
        try:
            prefill = int(float(r.get("prefill_length")))
            chunk = int(float(r.get("best_chunk_size")))
        except Exception:
            continue
        rows.append(
            {
                "model_id": str(r.get("model_id", "")),
                "revision": rev,
                "prefill_length": prefill,
                "best_chunk_size": chunk,
            }
        )
    if not rows:
        return

    per_model: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        model = str(r["model_id"])
        per_model.setdefault(model, []).append((int(r["prefill_length"]), int(r["best_chunk_size"])))

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("tab20")
    for idx, (model, points) in enumerate(sorted(per_model.items())):
        pts = sorted(points, key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.4,
            alpha=0.65,
            color=cmap(idx % 20),
            label=model,
        )

    all_prefills = sorted({int(r["prefill_length"]) for r in rows})
    ax.set_xticks(all_prefills)
    ax.set_title(f"Best Chunk Size vs Prefill Length ({mode}" + (f", revision={revision}" if revision else "") + ")")
    ax.set_xlabel("prefill_length")
    ax.set_ylabel("best_chunk_size")
    ax.grid(True, linestyle="--", alpha=0.3)
    if len(per_model) <= 12:
        ax.legend(loc="best", fontsize=8, title="model_id")
    else:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=7,
            title="model_id",
        )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _rebuild_outputs(results_dir: Path, records: list[dict[str, Any]], core_modes: list[str]) -> None:
    measurement_rows, best_rows = _collect_rows(records)
    _write_csv(results_dir / "all_measurements.csv", measurement_rows)
    _write_csv(results_dir / "best_chunks.csv", best_rows)
    prefill_lengths = sorted(
        {int(r["prefill_length"]) for r in measurement_rows if isinstance(r.get("prefill_length"), int)}
    )
    for mode in core_modes:
        for prefill_length in prefill_lengths:
            revisions = sorted(
                {
                    str(r.get("revision", "") or "")
                    for r in measurement_rows
                    if str(r.get("core_mode")) == mode and int(r.get("prefill_length", -1)) == int(prefill_length)
                }
            )
            for revision in revisions:
                if not revision:
                    continue
                _plot_mode_prefill_2d_chart(
                    mode,
                    int(prefill_length),
                    measurement_rows,
                    results_dir / f"prefill_tps_{mode}_prefill{prefill_length}_{_safe_filename(revision)}.png",
                    revision=revision,
                )
        best_revisions = sorted(
            {str(r.get("revision", "") or "") for r in best_rows if str(r.get("core_mode")) == mode}
        )
        for revision in best_revisions:
            if not revision:
                continue
            _plot_best_chunk_chart(
                mode,
                best_rows,
                results_dir / f"best_chunk_{mode}_{_safe_filename(revision)}.png",
                revision=revision,
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search best chunk_size for prefill TPS across models and core modes.",
    )
    parser.add_argument(
        "--mxq-dir",
        type=str,
        default=None,
        help="directory containing mxq files named as <model_without_group>-<W8|W4V8>.mxq",
    )
    parser.add_argument(
        "--core-modes",
        type=_parse_core_modes,
        default=_parse_core_modes("single,global4,global8"),
        help="comma-separated core modes",
    )
    parser.add_argument(
        "--prefill-lengths",
        type=_parse_int_csv,
        default=_parse_int_csv("1024,2048"),
        help="comma-separated prefill lengths (default: 1024,2048)",
    )
    parser.add_argument(
        "--decode-length", type=_parse_positive_int, default=16, help="fixed decode length per measurement"
    )
    parser.add_argument("--warmup", type=_parse_positive_int, default=1, help="warmup runs per model/core mode")
    parser.add_argument("--repeat", type=_parse_positive_int, default=3, help="repeat count per chunk size")
    parser.add_argument(
        "--time-guard-sec",
        type=float,
        default=300.0,
        help="if a single run takes longer than this, larger chunks are skipped (default: 300s)",
    )
    parser.add_argument(
        "--chunk-candidates",
        type=_parse_int_csv,
        default=_parse_int_csv("128,256,512,1024,2048"),
        help="comma-separated chunk sizes to test (default: 128,256,512,1024,2048)",
    )
    _add_pipeline_device_args(parser, device_default=None, trust_remote_code_default=True)
    parser.add_argument(
        "--skip-existing", action="store_true", help="reuse existing record JSON for model/core-mode pairs"
    )
    parser.add_argument(
        "--rebuild-charts",
        "--rebuild-chart",
        dest="rebuild_charts",
        action="store_true",
        help="skip benchmarking and rebuild CSV/charts from existing records",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="output directory (default: benchmark/transformers/results/prefill_chunk_search)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    if float(args.time_guard_sec) <= 0:
        print("--time-guard-sec must be > 0")
        return 2

    script_dir = Path(__file__).resolve().parent
    results_dir = (
        Path(args.results_dir).resolve() if args.results_dir else script_dir / "results" / "prefill_chunk_search"
    )
    records_dir = results_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    if args.rebuild_charts:
        for record_path in sorted(records_dir.glob("*.json")):
            loaded = _load_record(record_path)
            if loaded is not None:
                records.append(loaded)
        _rebuild_outputs(results_dir, records, args.core_modes)
        print(f"Rebuilt outputs from existing records: {results_dir}")
        return 0

    if not args.rebuild_charts:
        if not args.mxq_dir:
            print("--mxq-dir is required unless --rebuild-charts is used.")
            return 2
        mxq_dir = Path(args.mxq_dir).resolve()
        if not mxq_dir.is_dir():
            print(f"--mxq-dir does not exist or is not a directory: {mxq_dir}")
            return 2
    else:
        mxq_dir = None

    available = list_models(tasks="text-generation").get("text-generation", [])
    if not available:
        print("No text-generation models found.")
        return 0

    skipped_mxq_files: list[dict[str, str]] = []
    targets: list[dict[str, Any]] = []
    if mxq_dir is not None:
        targets, skipped_mxq_files = _discover_targets_from_mxq_dir(
            mxq_dir=mxq_dir,
            available_model_ids=[str(x) for x in available],
        )
        if not targets:
            print("No valid mxq targets found.")
            for row in skipped_mxq_files:
                print(f"[skip] {row['mxq_file']}: {row['reason']}")
            summary_payload = {
                "mxq_dir": str(mxq_dir),
                "total_mxq_files": len(list(mxq_dir.glob("*.mxq"))),
                "valid_targets": 0,
                "skipped_files": skipped_mxq_files,
            }
            _write_json(results_dir / "summary.json", summary_payload)
            return 0

    prefill_lengths = [int(x) for x in args.prefill_lengths]
    chunk_candidates = [int(x) for x in args.chunk_candidates]
    print(f"Valid mxq targets: {len(targets)}")
    print(f"Core modes: {args.core_modes}")
    print(f"Prefill lengths ({len(prefill_lengths)}): {prefill_lengths}")
    print(f"Chunk candidates ({len(chunk_candidates)}): {chunk_candidates}")
    print(f"Time guard: {float(args.time_guard_sec):.1f}s")
    if skipped_mxq_files:
        print(f"Skipped mxq files during discovery: {len(skipped_mxq_files)}")
        for row in skipped_mxq_files:
            print(f"[skip] {row['mxq_file']}: {row['reason']}")

    total = len(targets) * len(args.core_modes)
    pbar = tqdm(total=total, desc="model/core_mode search", unit="pair")
    try:
        for target in targets:
            model_id = str(target["model_id"])
            revision = str(target["revision"])
            mxq_path = str(target["mxq_path"])
            mxq_file = str(target["mxq_file"])
            for core_mode in args.core_modes:
                base_name = f"{_safe_filename(model_id)}__{revision}__{core_mode}.json"
                record_path = records_dir / base_name
                if args.skip_existing and record_path.is_file():
                    loaded = _load_record(record_path)
                    if loaded is not None:
                        records.append(loaded)
                        pbar.update(1)
                        continue

                if _is_cuda_device(args.device):
                    _clear_cuda_memory(args.device)

                pipeline_obj = None
                record: dict[str, Any] = {
                    "model_id": model_id,
                    "revision": revision,
                    "core_mode": core_mode,
                    "mxq_file": mxq_file,
                    "mxq_path": mxq_path,
                    "status": "failed",
                    "error": None,
                    "prefill_lengths": prefill_lengths,
                    "decode_length": int(args.decode_length),
                    "repeat": int(args.repeat),
                    "chunk_candidates": chunk_candidates,
                    "time_guard_sec": float(args.time_guard_sec),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                try:
                    pair_label = f"{model_id.split('/')[-1]}:{core_mode}"
                    pipeline_obj = _build_pipeline(
                        model_id=model_id,
                        revision=revision,
                        core_mode=core_mode,
                        device=args.device,
                        device_map=args.device_map,
                        dtype=args.dtype,
                        trust_remote_code=args.trust_remote_code,
                        mxq_path=mxq_path,
                    )
                    measurer = TPSMeasurer(pipeline_obj)
                    warmup_runs = max(0, int(args.warmup))
                    if warmup_runs > 0:
                        for _ in tqdm(
                            range(warmup_runs),
                            desc=f"warmup {pair_label}",
                            unit="run",
                            leave=False,
                        ):
                            measurer.measure(
                                num_prefill=int(prefill_lengths[0]),
                                num_decode=args.decode_length,
                                chunk_size=int(chunk_candidates[0]),
                                show_progress=False,
                            )
                    search_budget = max(1, len(prefill_lengths) * len(chunk_candidates))
                    with tqdm(
                        total=search_budget,
                        desc=f"search {pair_label}",
                        unit="eval",
                        leave=False,
                    ) as search_pbar:
                        searched = _run_prefill_grid(
                            measurer=measurer,
                            prefill_lengths=prefill_lengths,
                            chunk_candidates=chunk_candidates,
                            decode_length=args.decode_length,
                            repeat=args.repeat,
                            time_guard_sec=float(args.time_guard_sec),
                            progress_bar=search_pbar,
                            log_prefix=f"{model_id} [{core_mode}]",
                        )

                    record.update(searched)
                    record["status"] = "ok"
                    print(
                        f"[{model_id}] mode={core_mode} "
                        f"best_chunk={record['best_chunk_size']} "
                        f"best_median_prefill_tps={record['best_median_prefill_tps']:.4f}"
                    )
                except Exception as e:
                    record["status"] = "failed"
                    record["error"] = str(e)
                    print(f"Failed [{model_id}] mode={core_mode}: {e}")
                finally:
                    _release_pipeline(pipeline_obj, args.device)

                _write_json(record_path, record)
                records.append(record)
                pbar.update(1)
    finally:
        pbar.close()

    _rebuild_outputs(results_dir, records, args.core_modes)
    if skipped_mxq_files:
        _write_csv(results_dir / "skipped_mxq_files.csv", skipped_mxq_files)
    failed_pairs = [
        {
            "mxq_file": str(r.get("mxq_file", "")),
            "model_id": str(r.get("model_id", "")),
            "revision": str(r.get("revision", "")),
            "core_mode": str(r.get("core_mode", "")),
            "error": str(r.get("error", "")),
        }
        for r in records
        if str(r.get("status")) == "failed"
    ]
    if failed_pairs:
        _write_csv(results_dir / "failed_pairs.csv", failed_pairs)
    summary_payload = {
        "mxq_dir": None if mxq_dir is None else str(mxq_dir),
        "total_mxq_files": (0 if mxq_dir is None else len(list(mxq_dir.glob("*.mxq")))),
        "valid_targets": len(targets),
        "processed_pairs": total,
        "processed_records": len(records),
        "skipped_files_count": len(skipped_mxq_files),
        "skipped_files": skipped_mxq_files,
        "failed_pairs_count": len(failed_pairs),
        "failed_pairs": failed_pairs,
    }
    _write_json(results_dir / "summary.json", summary_payload)
    print(f"Saved outputs to: {results_dir}")
    print(f"Summary: {results_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
