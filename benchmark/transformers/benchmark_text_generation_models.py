import argparse
import json
import os
from dataclasses import asdict
from typing import Tuple

from transformers import pipeline as hf_pipeline

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
        ),
        decode_sweep=SweepData(
            x_values=decode.get("x_values", []),
            tps_values=decode.get("tps_values", []),
            time_values=decode.get("time_values", []),
        ),
    )


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
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")

    available = list_models(tasks="text-generation")
    model_ids = available.get("text-generation", [])
    if not model_ids:
        print("No text-generation models found.")
        return 0

    results_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        "text_generation",
    )
    os.makedirs(results_dir, exist_ok=True)

    for model_id in model_ids:
        print(f"=== {model_id} ===")
        base = _safe_filename(model_id)
        json_path = os.path.join(results_dir, f"{base}.json")
        png_path = os.path.join(results_dir, f"{base}.png")
        if (
            args.skip_existing
            and os.path.isfile(json_path)
            and os.path.isfile(png_path)
        ):
            print("Skipping (results exist).")
            continue
        pipeline = _build_pipeline(
            model_id,
            revision=args.revision,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        measurer = TPSMeasurer(pipeline)
        result = measurer.measure_full(
            prefill_range=args.prefill_range,
            decode_range=args.decode_range,
            fixed_decode_len=args.fixed_decode,
            fixed_prefill_len=args.fixed_prefill,
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        measurer.plot_and_save(result, save_path=png_path)

        del pipeline

    combined_results = []
    combined_labels = []
    combined_rows = []
    for model_id in model_ids:
        base = _safe_filename(model_id)
        json_path = os.path.join(results_dir, f"{base}.json")
        if not os.path.isfile(json_path):
            continue
        result = _load_result(json_path)
        combined_results.append(result)
        combined_labels.append(model_id)
        combined_rows.extend(list(BenchmarkResult.iter_rows(model_id, result)))

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
