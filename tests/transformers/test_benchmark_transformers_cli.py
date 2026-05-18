import json
import sys
from pathlib import Path

import pytest

_TRANSFORMERS_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "transformers"
if str(_TRANSFORMERS_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMERS_BENCHMARK_DIR))

from benchmark.transformers import benchmark_image_text_to_text_models as vlm_bench  # noqa: E402
from benchmark.transformers import benchmark_text_generation_models as text_bench  # noqa: E402


def test_text_benchmark_requires_subcommand() -> None:
    """Verify text benchmark rejects legacy no-subcommand invocations."""
    parser = text_bench._build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_text_benchmark_measure_defaults() -> None:
    """Verify text benchmark measure defaults match the TPS CLI."""
    args = text_bench._build_arg_parser().parse_args(["measure"])

    assert args.prefill == 128
    assert args.decode == 32
    assert args.repeat == 1
    assert args.warmup == 1
    assert args.core_mode is None
    assert args.prefill_chunk_size is None


def test_text_benchmark_sweep_defaults() -> None:
    """Verify text benchmark sweep defaults match the TPS CLI."""
    args = text_bench._build_arg_parser().parse_args(["sweep"])

    assert args.prefill_range == (512, 2048, 512)
    assert args.cache_lengths == [128, 512, 1024, 2048]
    assert args.decode_window == 32
    assert args.core_mode is None


def test_vlm_benchmark_requires_subcommand() -> None:
    """Verify VLM benchmark rejects legacy no-subcommand invocations."""
    parser = vlm_bench._build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_vlm_benchmark_measure_defaults() -> None:
    """Verify VLM benchmark measure defaults match the TPS CLI."""
    args = vlm_bench._build_arg_parser().parse_args(["measure"])

    assert args.image_resolution == 224
    assert args.prefill == 128
    assert args.decode == 32
    assert args.repeat == 1
    assert args.warmup == 1
    assert args.core_mode is None
    assert args.prompt == "Describe the image in one sentence."


def test_vlm_benchmark_sweep_defaults_and_removed_old_names() -> None:
    """Verify VLM sweep defaults and that old llm-prefixed options are rejected."""
    parser = vlm_bench._build_arg_parser()
    args = parser.parse_args(["sweep"])

    assert args.image_resolutions == [224, 384, 512, 768]
    assert args.llm_resolution is None
    assert args.prefill_range == (512, 2048, 512)
    assert args.cache_lengths == [128, 512, 1024, 2048]
    assert args.decode_window == 32

    with pytest.raises(SystemExit):
        parser.parse_args(["sweep", "--llm-prefill-range", "128:512:128"])


def test_text_measure_rebuild_outputs(tmp_path) -> None:
    """Verify text measure rebuild creates combined files from synthetic JSON."""
    payload = {
        "model": "model-a",
        "benchmark_type": "measure",
        "task": "text-generation",
        "prefill": 128,
        "decode": 32,
        "repeat": 1,
        "summary": {
            "prefill_tps": {"mean": 10.0},
            "decode_tps": {"mean": 20.0},
            "ttft_ms": {"mean": 30.0},
            "decode_duration_ms": {"mean": 40.0},
            "total_time_ms": {"mean": 70.0},
        },
        "device": None,
    }
    (tmp_path / "model-a_measure.json").write_text(json.dumps(payload), encoding="utf-8")

    text_bench._rebuild_measure_outputs(tmp_path)

    assert (tmp_path / "combined_measure.csv").is_file()
    assert (tmp_path / "combined_measure.md").is_file()


def test_vlm_measure_rebuild_outputs(tmp_path) -> None:
    """Verify VLM measure rebuild creates combined files from synthetic JSON."""
    payload = {
        "model": "vlm-a",
        "benchmark_type": "measure",
        "task": "image-text-to-text",
        "image_resolution": 224,
        "prefill": 128,
        "decode": 32,
        "repeat": 1,
        "summary": {
            "vision_encode_ms": {"mean": 1.0},
            "vision_fps": {"mean": 2.0},
            "llm_prefill_tps": {"mean": 3.0},
            "llm_decode_tps": {"mean": 4.0},
            "llm_ttft_ms": {"mean": 5.0},
            "llm_decode_duration_ms": {"mean": 6.0},
        },
        "device": None,
    }
    (tmp_path / "vlm-a_measure.json").write_text(json.dumps(payload), encoding="utf-8")

    vlm_bench._rebuild_measure_outputs(tmp_path)

    assert (tmp_path / "combined_measure.csv").is_file()
    assert (tmp_path / "combined_measure.md").is_file()
