import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Iterable, Tuple

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
        for _ in range(args.warmup):
            measurer.measure(
                num_prefill=args.fixed_prefill,
                num_decode=args.fixed_decode,
                trace_path=None,
            )
        result = measurer.measure_full(
            prefill_range=args.prefill_range,
            decode_range=args.decode_range,
            fixed_decode_len=args.fixed_decode,
            fixed_prefill_len=args.fixed_prefill,
        )
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

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        measurer.plot_and_save(result, save_path=png_path)

        del pipeline

    combined_results = []
    combined_labels = []
    combined_rows = []
    for _, _, label, base in targets:
        json_path = os.path.join(results_dir, f"{base}.json")
        if not os.path.isfile(json_path):
            continue
        result = _load_result(json_path)
        combined_results.append(result)
        combined_labels.append(label)
        combined_rows.extend(list(BenchmarkResult.iter_rows(label, result)))

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
