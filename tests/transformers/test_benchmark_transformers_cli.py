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

    assert args.batch_mode == "non_batch"
    assert args.prefill == 128
    assert args.decode == 32
    assert args.repeat == 1
    assert args.warmup == 1
    assert args.core_mode is None
    assert args.prefill_chunk_size is None


def test_text_benchmark_sweep_defaults() -> None:
    """Verify text benchmark sweep defaults match the TPS CLI."""
    args = text_bench._build_arg_parser().parse_args(["sweep"])

    assert args.batch_mode == "non_batch"
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

    assert args.batch_mode == "non_batch"
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

    assert args.batch_mode == "non_batch"
    assert args.image_resolutions == [224, 384, 512, 768]
    assert args.llm_resolution is None
    assert args.prefill_range == (512, 2048, 512)
    assert args.cache_lengths == [128, 512, 1024, 2048]
    assert args.decode_window == 32

    with pytest.raises(SystemExit):
        parser.parse_args(["sweep", "--llm-prefill-range", "128:512:128"])


@pytest.mark.parametrize("module", [text_bench, vlm_bench])
@pytest.mark.parametrize("command", ["measure", "sweep"])
def test_benchmark_batch_flags(module, command) -> None:
    """Verify benchmark scripts parse mutually exclusive batch target flags."""
    parser = module._build_arg_parser()

    assert parser.parse_args([command, "--batch"]).batch_mode == "batch"
    assert parser.parse_args([command, "--non-batch"]).batch_mode == "non_batch"
    with pytest.raises(SystemExit):
        parser.parse_args([command, "--batch", "--non-batch"])


def test_text_target_filtering_by_batch_mode(monkeypatch) -> None:
    """Verify text targets are filtered by resolved max_batch_size and GGUF artifacts."""
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = [
        ("mobilint/non-batch", [None], "non-batch", "non-batch", None),
        ("mobilint/batch", [None], "batch", "batch", None),
        ("mobilint/gguf", [None], "gguf", "gguf", None),
    ]

    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: model_id.endswith("gguf"))
    monkeypatch.setattr(
        text_bench,
        "_resolve_config_max_batch_size",
        lambda model_id, revision, *, task: 4 if model_id.endswith("batch") and not model_id.endswith("non-batch") else 1,
    )

    non_batch = text_bench._filter_text_targets_by_batch_mode(raw_targets, batch_mode="non_batch")
    batch = text_bench._filter_text_targets_by_batch_mode(raw_targets, batch_mode="batch")

    assert [target.model_id for target in non_batch] == ["mobilint/non-batch"]
    assert [target.model_id for target in batch] == ["mobilint/batch"]
    assert batch[0].max_batch_size == 4


def test_vlm_target_filtering_uses_image_text_task(monkeypatch, tmp_path) -> None:
    """Verify VLM target collection uses image-text-to-text batch metadata."""
    args = vlm_bench._build_arg_parser().parse_args([
        "measure",
        "--batch",
        "--results-dir",
        str(tmp_path),
    ])

    monkeypatch.setattr(vlm_bench, "list_models", lambda tasks: {"image-text-to-text": ["mobilint/vlm-a"]})

    def _fake_filter(raw_targets, *, batch_mode: str, task: str):
        assert batch_mode == "batch"
        assert task == "image-text-to-text"
        return [
            text_bench.TextBenchmarkTarget(
                model_id="mobilint/vlm-a",
                revision_candidates=[None],
                label="mobilint/vlm-a",
                base="mobilint_vlm-a",
                mxq_path=None,
                max_batch_size=2,
            )
        ]

    monkeypatch.setattr(vlm_bench, "_filter_text_targets_by_batch_mode", _fake_filter)

    _, _, run_targets = vlm_bench._collect_vlm_run_targets(args)

    assert len(run_targets) == 1
    assert run_targets[0][-1] == 2


def test_text_benchmark_resolves_mobilint_backend_per_target() -> None:
    """Verify Mobilint targets use NPU metrics even when the initial command has no model."""
    args = text_bench._build_arg_parser().parse_args(["measure", "--all"])

    text_bench._resolve_runtime_defaults(args, ["measure", "--all"])

    mobilint_args = text_bench._args_for_target_device_backend(args, model_id="mobilint/model-a")
    other_args = text_bench._args_for_target_device_backend(args, model_id="other/model-a")

    assert mobilint_args.device_backend == "npu"
    assert other_args.device_backend == "none"


def test_vlm_benchmark_resolves_mobilint_backend_per_target() -> None:
    """Verify VLM Mobilint targets use NPU metrics even when the initial command has no model."""
    args = vlm_bench._build_arg_parser().parse_args(["measure", "--all"])

    vlm_bench._resolve_runtime_defaults(args, ["measure", "--all"])

    mobilint_args = vlm_bench._args_for_target_device_backend(args, model_id="mobilint/model-a")
    other_args = vlm_bench._args_for_target_device_backend(args, model_id="other/model-a")

    assert mobilint_args.device_backend == "npu"
    assert other_args.device_backend == "none"


def test_benchmark_target_backend_preserves_explicit_backend() -> None:
    """Verify explicit device backend choices still override target policy."""
    args = text_bench._build_arg_parser().parse_args(["measure", "--all", "--device-backend", "gpu"])

    text_bench._resolve_runtime_defaults(args, ["measure", "--all", "--device-backend", "gpu"])
    target_args = text_bench._args_for_target_device_backend(args, model_id="mobilint/model-a")

    assert target_args.device_backend == "gpu"


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
