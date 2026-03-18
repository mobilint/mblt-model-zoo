import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, Iterable, Optional, Sequence, Tuple

from transformers import pipeline as hf_pipeline
from tqdm import tqdm

from mblt_model_zoo.hf_transformers.utils import list_models
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
    BenchmarkResult,
    SweepData,
    TPSMeasurer,
)


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
    try:
        value = int(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError("value must be an integer") from e
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def _parse_int_list_optional(spec: str | None) -> list[int] | None:
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


def _build_pipeline(
    model_id: str,
    revision: str | None = None,
    device: str = "cpu",
    device_map: str | None = None,
    dtype: str | None = None,
    trust_remote_code: bool = True,
):
    kwargs = {
        "task": "text-generation",
        "model": model_id,
        "trust_remote_code": trust_remote_code,
        "device": device,
    }
    if revision:
        kwargs["revision"] = revision
    if device_map:
        kwargs["device_map"] = device_map
    if dtype:
        kwargs["dtype"] = dtype
        try:
            return hf_pipeline(**kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = dtype
            return hf_pipeline(**kwargs)
    return hf_pipeline(**kwargs)


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


def _load_power(path: str) -> dict[str, float | None] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    power = payload.get("power")
    if not isinstance(power, dict):
        return None
    out: dict[str, float | None] = {}
    for key in (
        "avg_power_w",
        "p99_power_w",
        "total_energy_j",
        "prefill_tps_last",
        "decode_tps_last",
        "prefill_tok_per_j_last",
        "decode_tok_per_j_last",
        "prefill_j_per_tok_last",
        "decode_j_per_tok_last",
    ):
        value = power.get(key)
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
) -> Iterable[tuple[str, list[str | None], str, str]]:
    if not all_revisions:
        for model_id in model_ids:
            label = model_id
            base = _safe_filename(model_id)
            yield model_id, [revision], label, base
        return

    revision_map = [
        (["W8"], "-W8"),
        (["W4V8"], "-W4V8"),
    ]
    for model_id in model_ids:
        for revs, suffix in revision_map:
            label = f"{model_id}{suffix}"
            base = f"{_safe_filename(model_id)}{suffix}"
            yield model_id, revs, label, base


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


def _infer_gpu_ids(device: str, power_gpu_id: Optional[list[int]]) -> Optional[int | list[int]]:
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
        for name in ("model", "language_model", "vision_model", "text_model", "encoder", "decoder"):
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
            print(f"[power] failed to initialize NPU tracker: {e}")
            return None

    if backend == "gpu":
        from mblt_model_zoo.utils.power_tracker_gpu import GPUPowerTracker

        gpu_id = _infer_gpu_ids(args.device, args.power_gpu_id)
        try:
            return GPUPowerTracker(interval=args.power_interval, gpu_id=gpu_id)
        except Exception as e:
            print(f"[power] failed to initialize GPU tracker: {e}")
            return None

    return None


def _safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def _print_power_status(args: argparse.Namespace, tracker: Any) -> None:
    if not args.power:
        print("[power] disabled by --no-power")
        return
    if tracker is None:
        print("[power] enabled but no compatible tracker initialized (auto detection fallback)")
        return
    print(f"[power] enabled with {tracker.__class__.__name__} (interval={args.power_interval}s)")


def _write_power_combined_csv(path: str, rows: Sequence[dict[str, float | str | None]]) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "avg_power_w",
                "p99_power_w",
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


def _write_power_combined_markdown(path: str, rows: Sequence[dict[str, float | str | None]]) -> None:
    if not rows:
        return
    lines = [
        "| model | avg_power_w | p99_power_w | total_energy_j | prefill_tps_last | decode_tps_last | prefill_tok_per_j_last | decode_tok_per_j_last | prefill_j_per_tok_last | decode_j_per_tok_last |\n",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in rows:
        lines.append(
            "| {model} | {avg_power_w} | {p99_power_w} | {total_energy_j} | {prefill_tps_last} | {decode_tps_last} | {prefill_tok_per_j_last} | {decode_tok_per_j_last} | {prefill_j_per_tok_last} | {decode_j_per_tok_last} |\n".format(
                model=row["model"],
                avg_power_w="" if row["avg_power_w"] is None else f"{row['avg_power_w']:.6f}",
                p99_power_w="" if row["p99_power_w"] is None else f"{row['p99_power_w']:.6f}",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark text-generation models.")
    parser.add_argument(
        "--device",
        default="cpu",
        help='pipeline device (e.g., "cpu", "cuda:0")',
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help='pipeline device_map (e.g., "auto")',
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help='dtype for pipeline (e.g., "float16", "bfloat16")',
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to trust remote code when loading from HF",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="model revision (e.g., W8)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="benchmark W8 and W4W8 revisions only (skip main)",
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
        type=int,
        default=10,
        help="fixed decode length used during prefill sweep",
    )
    parser.add_argument(
        "--fixed-prefill",
        type=int,
        default=128,
        help="fixed prefill length used during decode sweep",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip models with existing JSON+PNG outputs",
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
    parser.add_argument(
        "--power",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable power tracking (default: on, disable via --no-power)",
    )
    parser.add_argument(
        "--power-device",
        choices=["auto", "gpu", "npu"],
        default="auto",
        help="power backend selection (default: auto)",
    )
    parser.add_argument(
        "--power-interval",
        type=float,
        default=0.2,
        help="power sampling interval in seconds",
    )
    parser.add_argument(
        "--power-gpu-id",
        type=_parse_int_list_optional,
        default=None,
        help="comma-separated GPU ids for power tracking (e.g., 0,1)",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")

    available = list_models(tasks="text-generation")
    model_ids = available.get("text-generation", [])
    if not model_ids:
        print("No text-generation models found.")
        return 0
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

    results_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        "text_generation",
    )
    os.makedirs(results_dir, exist_ok=True)

    targets = list(
        _iter_targets(
            model_ids,
            revision=args.revision,
            all_revisions=args.all,
        )
    )
    for model_id, revision_candidates, label, base in tqdm(
        targets,
        desc="Benchmarking models",
        total=len(targets),
        unit="model",
    ):
        print(f"=== {label} ===")
        revision = _select_revision(model_id, revision_candidates)
        if args.all and revision is None:
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
        try:
            pipeline = _build_pipeline(
                model_id,
                revision=revision,
                device=args.device,
                device_map=args.device_map,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
            )
        except Exception as e:
            if args.all and _revision_exists(model_id, revision or "") is None:
                print(f"Skipping (failed to load revision {revision}): {e}")
            else:
                print(f"Skipping (failed to load model): {e}")
            continue
        measurer = TPSMeasurer(pipeline)
        tracker = _build_power_tracker(args, pipeline)
        _print_power_status(args, tracker)
        for i in tqdm(range(args.warmup), desc=f"{label} warmup", leave=False):
            measurer.measure(
                num_prefill=args.fixed_prefill,
                num_decode=args.fixed_decode,
                trace_path=None,
                show_progress=True,
                progress_desc=f"{label} warmup generate {i + 1}/{args.warmup}",
            )
        run_start_t = time.time()
        if tracker is not None:
            tracker.start()
        try:
            result = measurer.measure_full(
                prefill_range=args.prefill_range,
                decode_range=args.decode_range,
                fixed_decode_len=args.fixed_decode,
                fixed_prefill_len=args.fixed_prefill,
                show_progress=True,
                progress_prefix=label,
            )
        finally:
            if tracker is not None:
                tracker.stop()
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
        power_payload: dict[str, float | None] | None = None
        if tracker is not None:
            metric = tracker.get_power_metric()
            avg_power = metric.get("avg_power_w")
            p99_power = metric.get("p99_power_w")
            if avg_power is not None:
                avg_power = float(avg_power)
            if p99_power is not None:
                p99_power = float(p99_power)
            elapsed = time.time() - run_start_t
            total_energy = avg_power * elapsed if avg_power is not None else None
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
                _safe_div(prefill_last, avg_power)
                if prefill_last is not None and avg_power is not None
                else None
            )
            decode_tpj = (
                _safe_div(decode_last, avg_power)
                if decode_last is not None and avg_power is not None
                else None
            )
            power_payload = {
                "avg_power_w": avg_power,
                "p99_power_w": p99_power,
                "total_energy_j": total_energy,
                "prefill_tps_last": prefill_last,
                "decode_tps_last": decode_last,
                "prefill_tok_per_j_last": prefill_tpj,
                "decode_tok_per_j_last": decode_tpj,
                "prefill_j_per_tok_last": _safe_div(1.0, prefill_tpj) if prefill_tpj else None,
                "decode_j_per_tok_last": _safe_div(1.0, decode_tpj) if decode_tpj else None,
            }
            print(
                "Power/Efficiency: "
                f"avg_power={avg_power if avg_power is not None else 'n/a'}W "
                f"prefill_tok_per_j(last)={prefill_tpj if prefill_tpj is not None else 'n/a'} "
                f"decode_tok_per_j(last)={decode_tpj if decode_tpj is not None else 'n/a'}"
            )

        payload: dict[str, Any] = {
            "benchmark": asdict(result),
            "power": power_payload,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        measurer.plot_and_save(result, save_path=png_path)

        del pipeline

    combined_results = []
    combined_labels = []
    combined_rows = []
    combined_power_rows: list[dict[str, float | str | None]] = []
    for _, _, label, base in targets:
        json_path = os.path.join(results_dir, f"{base}.json")
        if not os.path.isfile(json_path):
            continue
        result = _load_result(json_path)
        combined_results.append(result)
        combined_labels.append(label)
        combined_rows.extend(list(BenchmarkResult.iter_rows(label, result)))
        power = _load_power(json_path)
        if power:
            combined_power_rows.append(
                {
                    "model": label,
                    "avg_power_w": power.get("avg_power_w"),
                    "p99_power_w": power.get("p99_power_w"),
                    "total_energy_j": power.get("total_energy_j"),
                    "prefill_tps_last": power.get("prefill_tps_last"),
                    "decode_tps_last": power.get("decode_tps_last"),
                    "prefill_tok_per_j_last": power.get("prefill_tok_per_j_last"),
                    "decode_tok_per_j_last": power.get("decode_tok_per_j_last"),
                    "prefill_j_per_tok_last": power.get("prefill_j_per_tok_last"),
                    "decode_j_per_tok_last": power.get("decode_j_per_tok_last"),
                }
            )

    if combined_results:
        combined_path = os.path.join(results_dir, "combined.png")
        TPSMeasurer.plot_and_save_results(
            combined_results,
            combined_labels,
            save_path=combined_path,
        )
        combined_csv = os.path.join(results_dir, "combined.csv")
        combined_md = os.path.join(results_dir, "combined.md")
        BenchmarkResult.write_combined_csv(combined_csv, combined_rows)
        BenchmarkResult.write_combined_markdown(combined_md, combined_rows)
        if combined_power_rows:
            power_csv = os.path.join(results_dir, "combined_power.csv")
            power_md = os.path.join(results_dir, "combined_power.md")
            _write_power_combined_csv(power_csv, combined_power_rows)
            _write_power_combined_markdown(power_md, combined_power_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
