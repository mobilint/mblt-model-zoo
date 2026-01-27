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


def _parse_range_env(name: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    raw = os.getenv(name)
    if not raw:
        return default
    sep = ":" if ":" in raw else ("," if "," in raw else None)
    if sep is None:
        return default
    parts = [p.strip() for p in raw.split(sep)]
    if len(parts) != 3:
        return default
    try:
        start, end, step = (int(p) for p in parts)
    except ValueError:
        return default
    if start <= 0 or end <= 0 or step <= 0 or start > end:
        return default
    return start, end, step


def _build_pipeline(model_id: str):
    device = os.getenv("MBLT_DEVICE", "cpu")
    device_map = os.getenv("MBLT_DEVICE_MAP")
    dtype = os.getenv("MBLT_DTYPE")
    trust_remote_code = os.getenv("MBLT_TRUST_REMOTE_CODE", "true").lower() != "false"

    kwargs = {
        "task": "text-generation",
        "model": model_id,
        "trust_remote_code": trust_remote_code,
        "device": device,
    }
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


def main() -> int:
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

    prefill_range = _parse_range_env("MBLT_PREFILL_RANGE", (128, 512, 128))
    decode_range = _parse_range_env("MBLT_DECODE_RANGE", (128, 512, 128))
    fixed_decode = int(os.getenv("MBLT_FIXED_DECODE", "10"))
    fixed_prefill = int(os.getenv("MBLT_FIXED_PREFILL", "128"))
    skip_existing = os.getenv("MBLT_SKIP_EXISTING", "false").lower() == "true"

    last_measurer = None
    for model_id in model_ids:
        print(f"=== {model_id} ===")
        base = _safe_filename(model_id)
        json_path = os.path.join(results_dir, f"{base}.json")
        png_path = os.path.join(results_dir, f"{base}.png")
        if skip_existing and os.path.isfile(json_path) and os.path.isfile(png_path):
            print("Skipping (results exist).")
            continue
        pipeline = _build_pipeline(model_id)
        measurer = TPSMeasurer(pipeline)
        result = measurer.measure_full(
            prefill_range=prefill_range,
            decode_range=decode_range,
            fixed_decode_len=fixed_decode,
            fixed_prefill_len=fixed_prefill,
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        measurer.plot_and_save(result, save_path=png_path)

        last_measurer = measurer
        del pipeline

    if last_measurer is not None:
        combined_results = []
        combined_labels = []
        for model_id in model_ids:
            base = _safe_filename(model_id)
            json_path = os.path.join(results_dir, f"{base}.json")
            if not os.path.isfile(json_path):
                continue
            combined_results.append(_load_result(json_path))
            combined_labels.append(model_id)
        if combined_results:
            combined_path = os.path.join(results_dir, "combined.png")
            last_measurer.plot_and_save_results(
                combined_results,
                combined_labels,
                save_path=combined_path,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
