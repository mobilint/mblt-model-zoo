import json
import sys
from pathlib import Path

import pytest

_TRANSFORMERS_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "transformers"
if str(_TRANSFORMERS_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMERS_BENCHMARK_DIR))

from benchmark.transformers import benchmark_automatic_speech_recognition_models as asr_bench  # noqa: E402
from benchmark.transformers import benchmark_image_text_to_text_models as vlm_bench  # noqa: E402
from benchmark.transformers import benchmark_text_generation_models as text_bench  # noqa: E402
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (  # noqa: E402
    resolve_default_device,
    resolve_default_device_backend,
    resolve_device_tracker_interval_sec,
)


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
    assert args.core_mode == "global8"
    assert args.prefill_chunk_size is None


def test_text_benchmark_sweep_defaults() -> None:
    """Verify text benchmark sweep defaults match the TPS CLI."""
    args = text_bench._build_arg_parser().parse_args(["sweep"])

    assert args.batch_mode == "non_batch"
    assert args.prefill_range == (512, 2048, 512)
    assert args.cache_lengths == [128, 512, 1024, 2048]
    assert args.decode_window == 32
    assert args.core_mode == "global8"
    assert args.debug_errors is False


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
    assert args.core_mode == "global8"
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


def test_vlm_warmup_llm_kwargs_are_lightweight() -> None:
    """Verify VLM warmup uses fixed lightweight LLM dimensions."""
    warmup_kwargs = vlm_bench._vlm_warmup_llm_kwargs()

    assert warmup_kwargs == {
        "prefill_range": (128, 128, 128),
        "cache_lengths": [128],
        "decode_window": 32,
    }


@pytest.mark.parametrize(
    ("core_mode", "expected"),
    [
        (
            "single",
            {
                "vision_core_mode": "single",
                "vision_target_cores": ["0:0"],
                "text_core_mode": "single",
                "text_target_cores": ["0:0"],
            },
        ),
        (
            "global4",
            {
                "vision_core_mode": "global4",
                "vision_target_clusters": [0],
                "text_core_mode": "global4",
                "text_target_clusters": [0],
            },
        ),
        (
            "global8",
            {
                "vision_core_mode": "global8",
                "vision_target_clusters": [0, 1],
                "text_core_mode": "global8",
                "text_target_clusters": [0, 1],
            },
        ),
    ],
)
def test_vlm_core_mode_kwargs_are_prefixed(core_mode: str, expected: dict[str, object]) -> None:
    """Verify VLM benchmark maps shared core mode to composite config kwargs."""
    model_kwargs = vlm_bench._apply_vlm_core_mode_model_kwargs({}, core_mode)

    assert model_kwargs == expected
    assert "core_mode" not in model_kwargs
    assert "target_cores" not in model_kwargs
    assert "target_clusters" not in model_kwargs


def test_vlm_core_mode_none_does_not_add_kwargs() -> None:
    """Verify omitted VLM core mode does not create empty prefixed kwargs."""
    assert vlm_bench._apply_vlm_core_mode_model_kwargs({}, None) == {}


def test_vlm_core_mode_can_omit_default_single_target_cores() -> None:
    """Verify VLM batch benchmarks can keep single-mode target cores unset."""
    model_kwargs = vlm_bench._apply_vlm_core_mode_model_kwargs(
        {},
        "single",
        default_single_target_cores=None,
    )

    assert model_kwargs == {
        "vision_core_mode": "single",
        "text_core_mode": "single",
    }


def test_vlm_revision_preflight_skips_missing_revision(monkeypatch) -> None:
    """Verify VLM preflight rejects revisions that do not exist on the Hub."""
    monkeypatch.setattr(vlm_bench, "_revision_exists", lambda model_id, revision: False)

    available, reason = vlm_bench._vlm_revision_artifacts_available("mobilint/vlm-a", "W8", None)

    assert available is False
    assert "revision 'W8'" in str(reason)


def test_vlm_revision_preflight_skips_missing_mxq_artifact(monkeypatch) -> None:
    """Verify VLM preflight rejects configs that reference missing MXQ files."""
    monkeypatch.setattr(vlm_bench, "_revision_exists", lambda model_id, revision: True)
    monkeypatch.setattr(
        vlm_bench,
        "_read_raw_config",
        lambda model_id, revision: {
            "vision_config": {"mxq_path": "missing-vision.mxq"},
            "text_config": {"mxq_path": "present-text.mxq"},
        },
    )
    monkeypatch.setattr(vlm_bench, "_list_repo_files", lambda model_id, revision: ["present-text.mxq"])

    available, reason = vlm_bench._vlm_revision_artifacts_available("mobilint/vlm-a", "W8", None)

    assert available is False
    assert "missing-vision.mxq" in str(reason)


def test_vlm_revision_preflight_allows_existing_mxq_artifacts(monkeypatch) -> None:
    """Verify VLM preflight accepts revisions when all referenced MXQ files exist."""
    monkeypatch.setattr(vlm_bench, "_revision_exists", lambda model_id, revision: True)
    monkeypatch.setattr(
        vlm_bench,
        "_read_raw_config",
        lambda model_id, revision: {
            "vision_config": {"mxq_path": "vision.mxq"},
            "text_config": {"mxq_path": "text.mxq"},
        },
    )
    monkeypatch.setattr(vlm_bench, "_list_repo_files", lambda model_id, revision: ["vision.mxq", "text.mxq"])

    available, reason = vlm_bench._vlm_revision_artifacts_available("mobilint/vlm-a", "W8", None)

    assert available is True
    assert reason is None


@pytest.mark.parametrize("module", [text_bench, vlm_bench])
@pytest.mark.parametrize("command", ["measure", "sweep"])
def test_benchmark_batch_flags(module, command) -> None:
    """Verify benchmark scripts parse mutually exclusive batch target flags."""
    parser = module._build_arg_parser()

    assert parser.parse_args([command, "--batch"]).batch_mode == "batch"
    assert parser.parse_args([command, "--non-batch"]).batch_mode == "non_batch"
    with pytest.raises(SystemExit):
        parser.parse_args([command, "--batch", "--non-batch"])


@pytest.mark.parametrize("module", [text_bench, vlm_bench])
@pytest.mark.parametrize("command", ["measure", "sweep"])
def test_benchmark_parser_accepts_npu_rail_metrics(module, command) -> None:
    """Verify benchmark subcommand parsers expose the NPU rail metric option."""
    args = module._build_arg_parser().parse_args([command, "--device-npu-rail-metrics", "all"])

    assert args.device_npu_rail_metrics == "all"


@pytest.mark.parametrize("module", [text_bench, vlm_bench])
@pytest.mark.parametrize("command", ["measure", "sweep"])
def test_benchmark_parser_defaults_npu_rail_metrics(module, command) -> None:
    """Verify benchmark subcommand parsers keep the default low-latency NPU rail."""
    args = module._build_arg_parser().parse_args([command])

    assert args.device_npu_rail_metrics == "npu"


def test_asr_benchmark_parser_accepts_npu_rail_metrics() -> None:
    """Verify the ASR benchmark parser exposes the shared NPU rail metric option."""
    args = asr_bench._parse_args(["--device-npu-rail-metrics", "npu,ddr"])

    assert args.device_npu_rail_metrics == ["npu", "ddr"]


def test_asr_benchmark_parser_defaults_npu_rail_metrics() -> None:
    """Verify the ASR benchmark parser keeps the default low-latency NPU rail."""
    args = asr_bench._parse_args([])

    assert args.device_npu_rail_metrics == "npu"


@pytest.mark.parametrize("module", [text_bench, vlm_bench])
@pytest.mark.parametrize("command", ["measure", "sweep"])
def test_benchmark_batch_defaults_to_single_core_mode(module, command) -> None:
    """Verify batch LLM benchmarks default to the only supported single core mode."""
    args = module._build_arg_parser().parse_args([command, "--batch"])

    module._resolve_runtime_defaults(args, [command, "--batch"])

    assert args.core_mode == "single"


@pytest.mark.parametrize("module", [text_bench, vlm_bench])
def test_benchmark_batch_mode_disables_default_single_target_cores(module) -> None:
    """Verify batch benchmark paths do not inject the implicit single target core."""
    batch_args = module._build_arg_parser().parse_args(["measure", "--batch"])
    non_batch_args = module._build_arg_parser().parse_args(["measure", "--non-batch"])

    assert module._default_single_target_cores_for_batch_mode(batch_args) is None
    assert module._default_single_target_cores_for_batch_mode(non_batch_args) == ("0:0",)


@pytest.mark.parametrize("module", [text_bench, vlm_bench])
@pytest.mark.parametrize("command", ["measure", "sweep"])
def test_benchmark_batch_rejects_non_single_core_mode(module, command) -> None:
    """Verify explicit non-single core modes are rejected for batch LLM benchmarks."""
    args = module._build_arg_parser().parse_args([command, "--batch", "--core-mode", "global8"])

    with pytest.raises(SystemExit, match="only supports --core-mode single"):
        module._resolve_runtime_defaults(args, [command, "--batch", "--core-mode", "global8"])


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
        lambda model_id, revision, *, task: (
            4 if model_id.endswith("batch") and not model_id.endswith("non-batch") else 1
        ),
    )

    non_batch = text_bench._filter_text_targets_by_batch_mode(raw_targets, batch_mode="non_batch")
    batch = text_bench._filter_text_targets_by_batch_mode(raw_targets, batch_mode="batch")

    assert [target.model_id for target in non_batch] == ["mobilint/non-batch"]
    assert [target.model_id for target in batch] == ["mobilint/batch"]
    assert batch[0].max_batch_size == 4


def test_text_target_filtering_treats_missing_max_batch_size_as_non_batch(monkeypatch) -> None:
    """Verify missing batch metadata keeps original targets only in non-batch mode."""
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = [
        ("upstream/original", [None], "original", "original", None),
    ]

    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: False)
    monkeypatch.setattr(text_bench, "_resolve_config_max_batch_size", lambda model_id, revision, *, task: None)

    non_batch = text_bench._filter_text_targets_by_batch_mode(raw_targets, batch_mode="non_batch")
    batch = text_bench._filter_text_targets_by_batch_mode(raw_targets, batch_mode="batch")

    assert [target.model_id for target in non_batch] == ["upstream/original"]
    assert non_batch[0].max_batch_size == 1
    assert batch == []


def test_vlm_target_filtering_uses_image_text_task(monkeypatch, tmp_path) -> None:
    """Verify VLM target collection uses image-text-to-text batch metadata."""
    args = vlm_bench._build_arg_parser().parse_args(
        [
            "measure",
            "--batch",
            "--output-dir",
            str(tmp_path),
        ]
    )

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
                batch_mode="batch",
            )
        ]

    monkeypatch.setattr(vlm_bench, "_filter_text_targets_by_batch_mode", _fake_filter)

    _, _, run_targets = vlm_bench._collect_vlm_run_targets(args)

    assert len(run_targets) == 1
    assert run_targets[0][-2:] == (2, "batch")


def test_vlm_measure_stops_tracker_when_vision_measure_fails(monkeypatch, tmp_path) -> None:
    """Verify VLM fixed measure stops the whole-run tracker when vision measurement fails."""
    args = vlm_bench._build_arg_parser().parse_args(
        [
            "measure",
            "--output-dir",
            str(tmp_path),
        ]
    )
    stopped: list[bool] = []

    class _FakeTracker:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            stopped.append(True)

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline) -> None:
            self._vision_calls = 0

        def measure_vision(self, *args, **kwargs):
            self._vision_calls += 1
            if self._vision_calls == 1:
                return [(0.1, 10.0)]
            raise RuntimeError("vision failed")

        def measure_llm_full(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        vlm_bench,
        "_collect_vlm_run_targets",
        lambda args: (tmp_path, False, [("model-a", None, "model-a", "model-a", None, None, 1)]),
    )
    monkeypatch.setattr(vlm_bench, "_collect_host_pc_info", lambda results_dir: None)
    monkeypatch.setattr(
        vlm_bench,
        "_vlm_revision_artifacts_available",
        lambda model_id, revision, mxq_path: (True, None),
    )
    monkeypatch.setattr(vlm_bench, "_build_pipeline", lambda *args, **kwargs: object())
    monkeypatch.setattr(vlm_bench, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)
    monkeypatch.setattr(vlm_bench, "_build_device_tracker", lambda args, pipeline: _FakeTracker())
    monkeypatch.setattr(vlm_bench, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(vlm_bench, "_release_pipeline", lambda pipeline, device: None)
    monkeypatch.setattr(vlm_bench, "_rebuild_measure_outputs", lambda results_dir: None)

    assert vlm_bench._run_measure(args) == 0
    assert stopped == [True]


def test_vlm_measure_batch_energy_uses_batch_vision_latency(monkeypatch, tmp_path) -> None:
    """Verify VLM fixed measure derives energy and image efficiency from the power trace."""
    args = vlm_bench._build_arg_parser().parse_args(
        [
            "measure",
            "--batch",
            "--output-dir",
            str(tmp_path),
            "--repeat",
            "1",
        ]
    )

    class _FakeTracker:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline) -> None:
            pass

        def measure_vision(self, *args, **kwargs):
            return [(0.1, 10.0)]

        def measure_llm_full(self, *args, **kwargs):
            return vlm_bench.BenchmarkResult(
                prefill_sweep=vlm_bench.SweepData(x_values=[128], tps_values=[20.0], time_values=[0.2]),
                decode_sweep=vlm_bench.SweepData(x_values=[128], tps_values=[40.0], time_values=[0.3]),
                prefill_phase_duration_s=0.2,
                decode_phase_duration_s=0.3,
            )

    monkeypatch.setattr(
        vlm_bench,
        "_collect_vlm_run_targets",
        lambda args: (tmp_path, False, [("model-a", None, "model-a", "model-a", None, None, 4)]),
    )
    monkeypatch.setattr(vlm_bench, "_collect_host_pc_info", lambda results_dir: None)
    monkeypatch.setattr(
        vlm_bench,
        "_vlm_revision_artifacts_available",
        lambda model_id, revision, mxq_path: (True, None),
    )
    monkeypatch.setattr(vlm_bench, "_build_pipeline", lambda *args, **kwargs: object())
    monkeypatch.setattr(vlm_bench, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)
    monkeypatch.setattr(vlm_bench, "_build_device_tracker", lambda args, pipeline: _FakeTracker())
    monkeypatch.setattr(vlm_bench, "_extract_device_metric", lambda tracker: {"avg_power_w": 10.0})
    monkeypatch.setattr(
        vlm_bench,
        "_extract_device_time_series",
        lambda tracker: {"power_w": [{"timestamp_s": 0.0, "value": 10.0}, {"timestamp_s": 0.9, "value": 10.0}]},
    )
    monkeypatch.setattr(vlm_bench, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(vlm_bench, "_release_pipeline", lambda pipeline, device: None)
    monkeypatch.setattr(vlm_bench, "_rebuild_measure_outputs", lambda results_dir: None)

    assert vlm_bench._run_measure(args) == 0

    payload = json.loads((tmp_path / "model-a_measure.json").read_text(encoding="utf-8"))
    assert payload["device"]["total_energy_j"] == pytest.approx(9.0)
    assert payload["device"]["vision_img_per_j"] == pytest.approx(4.0 / 9.0)


def test_text_benchmark_resolves_mobilint_backend_per_target() -> None:
    """Verify Mobilint targets use NPU metrics even when the initial command has no model."""
    args = text_bench._build_arg_parser().parse_args(["measure", "--all"])

    text_bench._resolve_runtime_defaults(args, ["measure", "--all"])

    mobilint_args = text_bench._args_for_target_device_backend(args, model_id="mobilint/model-a")
    other_args = text_bench._args_for_target_device_backend(args, model_id="other/model-a")

    assert mobilint_args.device == "cpu"
    assert mobilint_args.device_backend == "npu"
    assert other_args.device == "cuda"
    assert other_args.device_backend == "gpu"


def test_vlm_benchmark_resolves_mobilint_backend_per_target() -> None:
    """Verify VLM Mobilint targets use NPU metrics even when the initial command has no model."""
    args = vlm_bench._build_arg_parser().parse_args(["measure", "--all"])

    vlm_bench._resolve_runtime_defaults(args, ["measure", "--all"])

    mobilint_args = vlm_bench._args_for_target_device_backend(args, model_id="mobilint/model-a")
    other_args = vlm_bench._args_for_target_device_backend(args, model_id="other/model-a")

    assert mobilint_args.device == "cpu"
    assert mobilint_args.device_backend == "npu"
    assert other_args.device == "cuda"
    assert other_args.device_backend == "gpu"


def test_asr_benchmark_resolves_mobilint_backend_per_target() -> None:
    """Verify ASR Mobilint targets use NPU metrics even when the initial command has no model."""
    args = asr_bench._parse_args(["--all"])

    asr_bench._resolve_runtime_defaults(args, ["--all"])

    mobilint_args = asr_bench._args_for_target_device_backend(args, model_id="mobilint/model-a")
    other_args = asr_bench._args_for_target_device_backend(args, model_id="other/model-a")

    assert mobilint_args.device == "cpu"
    assert mobilint_args.device_backend == "npu"
    assert other_args.device == "cuda"
    assert other_args.device_backend == "gpu"


@pytest.mark.parametrize(
    ("model_id", "mxq_path", "mxq_dir", "expected_device", "expected_backend"),
    [
        ("mobilint/model-a", None, None, "cpu", "npu"),
        ("other/model-a", None, None, "cuda", "gpu"),
        ("other/model-a", "model.mxq", None, "cpu", "npu"),
        ("other/model-a", None, "mxq", "cpu", "npu"),
    ],
)
def test_benchmark_common_runtime_default_policy(
    model_id: str,
    mxq_path: str | None,
    mxq_dir: str | None,
    expected_device: str,
    expected_backend: str,
) -> None:
    """Verify shared benchmark runtime defaults are target-aware."""
    assert (
        resolve_default_device(
            device=None,
            device_explicit=False,
            model_id=model_id,
            mxq_path=mxq_path,
            mxq_dir=mxq_dir,
        )
        == expected_device
    )
    assert (
        resolve_default_device_backend(
            device_backend="gpu",
            device_backend_explicit=False,
            model_id=model_id,
            mxq_path=mxq_path,
            mxq_dir=mxq_dir,
        )
        == expected_backend
    )


def test_benchmark_common_runtime_default_policy_preserves_explicit_values() -> None:
    """Verify explicit device/backend values are not overwritten by target policy."""
    assert (
        resolve_default_device(
            device="cuda:1",
            device_explicit=True,
            model_id="mobilint/model-a",
        )
        == "cuda:1"
    )
    assert (
        resolve_default_device_backend(
            device_backend="gpu",
            device_backend_explicit=True,
            model_id="mobilint/model-a",
        )
        == "gpu"
    )


@pytest.mark.parametrize(("backend", "expected"), [("npu", 1.0), ("gpu", 0.1), ("cpu", 0.1)])
def test_benchmark_common_tracker_interval_policy(backend: str, expected: float) -> None:
    """Verify tracker sampling intervals are selected by resolved backend."""
    assert resolve_device_tracker_interval_sec(backend) == pytest.approx(expected)


def test_benchmark_target_backend_preserves_explicit_backend() -> None:
    """Verify explicit device backend choices still override target policy."""
    args = text_bench._build_arg_parser().parse_args(["measure", "--all", "--device-backend", "gpu"])

    text_bench._resolve_runtime_defaults(args, ["measure", "--all", "--device-backend", "gpu"])
    target_args = text_bench._args_for_target_device_backend(args, model_id="mobilint/model-a")

    assert target_args.device_backend == "gpu"


def test_benchmark_target_device_preserves_explicit_device() -> None:
    """Verify explicit device choices still override target device policy."""
    args = text_bench._build_arg_parser().parse_args(["measure", "--all", "--device", "cuda:1"])

    text_bench._resolve_runtime_defaults(args, ["measure", "--all", "--device", "cuda:1"])
    target_args = text_bench._args_for_target_device_backend(args, model_id="mobilint/model-a")

    assert target_args.device == "cuda:1"


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


def test_text_load_result_pads_missing_latency_arrays(tmp_path) -> None:
    """Verify old text sweep JSON without latency arrays still produces rows."""
    payload = {
        "model": "text-a",
        "benchmark": {
            "prefill_sweep": {"x_values": [8], "tps_values": [10.0], "time_values": [0.8]},
            "decode_sweep": {"x_values": [4], "tps_values": [20.0], "time_values": [0.2]},
        },
    }
    path = tmp_path / "text-a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = text_bench._load_result(str(path))
    rows = list(text_bench.BenchmarkResult.iter_rows("text-a", result))

    assert len(rows) == 2
    assert result.prefill_sweep.avg_total_token_latency_values == [None]
    assert result.decode_sweep.avg_npu_token_latency_values == [None]


def test_text_aggregate_results_tolerates_missing_latency_arrays() -> None:
    """Verify text repeated sweeps aggregate missing latency arrays and metadata."""
    first = text_bench.BenchmarkResult(
        prefill_sweep=text_bench.SweepData(x_values=[8], tps_values=[10.0], time_values=[0.8]),
        decode_sweep=text_bench.SweepData(x_values=[4], tps_values=[20.0], time_values=[0.2]),
        decode_prefill_modes=["fake"],
        prefill_phase_duration_s=0.8,
        decode_phase_duration_s=0.2,
    )
    second = text_bench.BenchmarkResult(
        prefill_sweep=text_bench.SweepData(x_values=[8], tps_values=[20.0], time_values=[1.2]),
        decode_sweep=text_bench.SweepData(x_values=[4], tps_values=[40.0], time_values=[0.4]),
        decode_prefill_modes=["fake"],
        prefill_phase_duration_s=1.2,
        decode_phase_duration_s=0.4,
    )

    result = text_bench._aggregate_benchmark_results([first, second])

    assert result.prefill_sweep.tps_values == [15.0]
    assert result.decode_sweep.tps_values == [30.0]
    assert result.prefill_sweep.avg_total_token_latency_values == [None]
    assert result.decode_prefill_modes == ["fake"]
    assert result.prefill_phase_duration_s == pytest.approx(1.0)
    assert result.decode_phase_duration_s == pytest.approx(0.3)


def test_vlm_aggregate_llm_runs_tolerates_missing_latency_arrays() -> None:
    """Verify old VLM LLM runs without latency arrays aggregate and emit rows."""
    runs = [
        {
            "prefill_sweep": {"x_values": [8], "tps_values": [10.0], "time_values": [0.8]},
            "decode_sweep": {"x_values": [4], "tps_values": [20.0], "time_values": [0.2]},
        },
        {
            "prefill_sweep": {"x_values": [8], "tps_values": [30.0], "time_values": [1.2]},
            "decode_sweep": {"x_values": [4], "tps_values": [40.0], "time_values": [0.4]},
        },
    ]

    result = vlm_bench._aggregate_vlm_llm_runs(runs)
    rows = list(vlm_bench.BenchmarkResult.iter_rows("vlm-a", result))

    assert result.prefill_sweep.tps_values == [20.0]
    assert result.decode_sweep.tps_values == [30.0]
    assert result.prefill_sweep.avg_total_token_latency_values == [None]
    assert len(rows) == 2
