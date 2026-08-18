import csv
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

_SKIP_TIMESTAMP_GAP_S = 3600.0


def _backdate(path: Path, *, seconds: float = _SKIP_TIMESTAMP_GAP_S) -> None:
    """Backdate a file's atime and mtime so reconciliation tests can express "on-disk is older"."""
    stamp = path.stat().st_mtime - seconds
    os.utime(path, (stamp, stamp))


def _sidecar_row_now(**fields: Any) -> dict[str, Any]:
    """Return a sidecar row stamped with ``recorded_at = time.time()`` for "sidecar is newer" tests."""
    return {**fields, "recorded_at": time.time()}


_TRANSFORMERS_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "transformers"
if str(_TRANSFORMERS_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMERS_BENCHMARK_DIR))

from benchmark.transformers import benchmark_automatic_speech_recognition_models as asr_bench  # noqa: E402
from benchmark.transformers import benchmark_image_text_to_text_models as vlm_bench  # noqa: E402
from benchmark.transformers import benchmark_text_generation_models as text_bench  # noqa: E402
from mblt_model_zoo.cli import tps as tps_cli  # noqa: E402
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
    assert args.npu_prefill_chunk_size is None


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

    monkeypatch.setattr(
        vlm_bench,
        "list_default_model_ids",
        lambda task, *, include_private=False: ["mobilint/vlm-a"],
    )

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
        lambda args: (tmp_path, False, [("model-a", None, "model-a", "model-a", None, None, 1, "non_batch")]),
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

    phase_tracker_runs: list[tuple[_FakeTracker, _FakeTracker]] = []

    def _fake_build_phase_trackers(args, pipeline):
        del args, pipeline
        trackers = (_FakeTracker(), _FakeTracker())
        phase_tracker_runs.append(trackers)
        return trackers

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
        lambda args: (tmp_path, False, [("model-a", None, "model-a", "model-a", None, None, 4, "batch")]),
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
    monkeypatch.setattr(vlm_bench, "_build_phase_trackers", _fake_build_phase_trackers)
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
    assert len(phase_tracker_runs) == 1

    payload = json.loads((tmp_path / "model-a_measure.json").read_text(encoding="utf-8"))
    assert payload["device"]["vision_energy_j"] == pytest.approx(9.0)
    assert payload["device"]["llm_prefill_energy_j"] == pytest.approx(9.0)
    assert payload["device"]["llm_decode_energy_j"] == pytest.approx(9.0)
    assert payload["device"]["llm_total_energy_j"] == pytest.approx(18.0)
    assert payload["device"]["total_energy_j"] == pytest.approx(27.0)
    assert payload["device"]["total_energy_j"] == pytest.approx(
        payload["device"]["vision_energy_j"] + payload["device"]["llm_total_energy_j"]
    )
    assert payload["device"]["vision_img_per_j"] == pytest.approx(4.0 / 9.0)


def test_vlm_measure_tps_per_w_scales_by_measured_repeat_count(monkeypatch, tmp_path) -> None:
    """Verify VLM fixed measure TPS/W uses all repeated runs included in total energy."""
    args = vlm_bench._build_arg_parser().parse_args(
        [
            "measure",
            "--batch",
            "--output-dir",
            str(tmp_path),
            "--repeat",
            "2",
            "--prefill",
            "128",
            "--decode",
            "32",
        ]
    )

    class _FakeTracker:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    phase_tracker_runs: list[tuple[_FakeTracker, _FakeTracker]] = []

    def _fake_build_phase_trackers(args, pipeline):
        del args, pipeline
        trackers = (_FakeTracker(), _FakeTracker())
        phase_tracker_runs.append(trackers)
        return trackers

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline) -> None:
            pass

        def measure_vision(self, *args, **kwargs):
            return [(0.1, 10.0)]

        def measure_llm_full(self, *args, **kwargs):
            return vlm_bench.BenchmarkResult(
                prefill_sweep=vlm_bench.SweepData(x_values=[128], tps_values=[20.0], time_values=[0.2]),
                decode_sweep=vlm_bench.SweepData(x_values=[128], tps_values=[40.0], time_values=[0.3]),
            )

    monkeypatch.setattr(
        vlm_bench,
        "_collect_vlm_run_targets",
        lambda args: (tmp_path, False, [("model-a", None, "model-a", "model-a", None, None, 4, "batch")]),
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
    monkeypatch.setattr(vlm_bench, "_build_phase_trackers", _fake_build_phase_trackers)
    monkeypatch.setattr(vlm_bench, "_extract_device_metric", lambda tracker: {"avg_power_w": 10.0})
    monkeypatch.setattr(
        vlm_bench,
        "_extract_device_time_series",
        lambda tracker: {"power_w": [{"timestamp_s": 0.0, "value": 10.0}, {"timestamp_s": 1.0, "value": 10.0}]},
    )
    monkeypatch.setattr(vlm_bench, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(vlm_bench, "_release_pipeline", lambda pipeline, device: None)
    monkeypatch.setattr(vlm_bench, "_rebuild_measure_outputs", lambda results_dir: None)

    assert vlm_bench._run_measure(args) == 0
    assert len(phase_tracker_runs) == 2
    assert phase_tracker_runs[0][0] is not phase_tracker_runs[1][0]
    assert phase_tracker_runs[0][1] is not phase_tracker_runs[1][1]

    payload = json.loads((tmp_path / "model-a_measure.json").read_text(encoding="utf-8"))
    assert payload["device"]["vision_energy_j"] == pytest.approx(20.0)
    assert payload["device"]["llm_prefill_energy_j"] == pytest.approx(20.0)
    assert payload["device"]["llm_decode_energy_j"] == pytest.approx(20.0)
    assert payload["device"]["llm_total_energy_j"] == pytest.approx(40.0)
    assert payload["device"]["total_energy_j"] == pytest.approx(60.0)


def test_vlm_sweep_token_helpers_use_whole_sweep_scope() -> None:
    """Verify VLM sweep token helpers match whole-sweep trace energy scope."""
    result = vlm_bench.BenchmarkResult(
        prefill_sweep=vlm_bench.SweepData(x_values=[128, 256], tps_values=[10.0, 20.0], time_values=[0.1, 0.2]),
        decode_sweep=vlm_bench.SweepData(
            x_values=[128, 256, 512],
            tps_values=[30.0, 40.0, 50.0],
            time_values=[0.3, 0.4, 0.5],
        ),
    )

    assert vlm_bench._sweep_prefill_token_count(result, batch_size=2) == (128 + 256) * 2
    assert vlm_bench._sweep_decode_token_count(result, decode_window=32, batch_size=2) == 32 * 3 * 2


def test_vlm_benchmark_sweep_populates_llm_tps_per_w(monkeypatch) -> None:
    """Verify VLM benchmark sweep derives phase efficiency from phase trace energy."""
    args = Namespace(
        llm_resolution=224,
        image_resolutions=[224],
        original_models=False,
        mxq_dir=None,
        npu_prefill_chunk_size=None,
        warmup=0,
        repeat=1,
        prompt="prompt",
        batch_size=2,
        prefill_range=(128, 256, 128),
        cache_lengths=[128, 256, 512],
        decode_window=32,
        batch_mode="batch",
    )

    class _FakeTracker:
        def start(self) -> None:
            pass

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline) -> None:
            pass

        def measure_vision(self, *args, **kwargs):
            return [(0.1, 10.0)]

        def measure_llm_full(self, *args, **kwargs):
            return vlm_bench.BenchmarkResult(
                prefill_sweep=vlm_bench.SweepData(x_values=[128, 256], tps_values=[10.0, 20.0], time_values=[0.1, 0.2]),
                decode_sweep=vlm_bench.SweepData(
                    x_values=[128, 256, 512],
                    tps_values=[30.0, 40.0, 50.0],
                    time_values=[0.3, 0.4, 0.5],
                ),
            )

    monkeypatch.setattr(vlm_bench, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)
    monkeypatch.setattr(vlm_bench, "_build_device_tracker", lambda args, pipeline: _FakeTracker())
    monkeypatch.setattr(vlm_bench, "_build_phase_trackers", lambda args, pipeline: (_FakeTracker(), _FakeTracker()))
    monkeypatch.setattr(vlm_bench, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(vlm_bench, "_stop_tracker_safe", lambda tracker: None)
    monkeypatch.setattr(vlm_bench, "_extract_device_metric", lambda tracker: {"avg_power_w": 10.0})
    monkeypatch.setattr(
        vlm_bench,
        "_extract_device_time_series",
        lambda tracker: {"power_w": [{"timestamp_s": 0.0, "value": 10.0}, {"timestamp_s": 1.0, "value": 10.0}]},
    )

    payload, rows = vlm_bench._run_model(args, "model-a", "model-a", object())

    llm_run = payload["benchmark"]["llm_results"]["runs"][0]
    llm_summary = payload["benchmark"]["llm_results"]["summary"]
    llm_rows = [row for row in rows if row["type"] == "llm"]
    assert llm_run["vision_energy_j"] == pytest.approx(10.0)
    assert llm_run["llm_prefill_energy_j"] == pytest.approx(10.0)
    assert llm_run["llm_decode_energy_j"] == pytest.approx(10.0)
    assert llm_run["llm_total_energy_j"] == pytest.approx(20.0)
    assert llm_run["total_energy_j"] == pytest.approx(30.0)
    assert llm_summary["llm_total_energy_j"]["mean"] == pytest.approx(20.0)
    assert llm_summary["total_energy_j"]["mean"] == pytest.approx(30.0)
    assert llm_run["prefill_tps_per_w"] == pytest.approx(((128 + 256) * 2) / 10.0)
    assert llm_run["decode_tps_per_w"] == pytest.approx((32 * 3 * 2) / 10.0)
    assert llm_run["prefill_j_per_token"] == pytest.approx(10.0 / ((128 + 256) * 2))
    assert llm_run["decode_j_per_token"] == pytest.approx(10.0 / (32 * 3 * 2))
    assert llm_summary["prefill_tps_per_w"]["mean"] == pytest.approx(llm_run["prefill_tps_per_w"])
    assert llm_summary["decode_tps_per_w"]["mean"] == pytest.approx(llm_run["decode_tps_per_w"])
    assert [row["prefill_tps_per_w"] for row in llm_rows] == [pytest.approx(llm_run["prefill_tps_per_w"])] * len(
        llm_rows
    )
    assert [row["decode_tps_per_w"] for row in llm_rows] == [pytest.approx(llm_run["decode_tps_per_w"])] * len(llm_rows)


def test_tps_cli_vlm_sweep_writes_phase_tps_per_w(monkeypatch, tmp_path) -> None:
    """Verify TPS CLI VLM sweep writes phase efficiency to JSON and CSV rows."""
    args = Namespace(
        task="image-text-to-text",
        model="model-a",
        tokenizer=None,
        device=None,
        device_backend=None,
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        base_embedding_path=None,
        draft_embedding_path=None,
        base_mxq_path=None,
        draft_mxq_path=None,
        fc_mxq_path=None,
        base_core_mode=None,
        draft_core_mode=None,
        fc_core_mode=None,
        base_target_cores=None,
        draft_target_cores=None,
        fc_target_cores=None,
        base_target_clusters=None,
        draft_target_clusters=None,
        fc_target_clusters=None,
        batch_size=2,
        image_resolutions=[224],
        llm_resolution=None,
        warmup=0,
        repeat=1,
        prompt="prompt",
        prefill_range=(128, 256, 128),
        cache_lengths=[128, 256, 512],
        decode_window=32,
        npu_prefill_chunk_size=None,
        device_metrics=True,
        json=str(tmp_path / "vlm.json"),
        csv=str(tmp_path / "vlm.csv"),
        plot=None,
    )

    class _FakeTracker:
        def start(self) -> None:
            pass

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline) -> None:
            pass

        def measure_vision(self, *args, **kwargs):
            return [(0.1, 10.0)]

        def measure_llm_full(self, *args, **kwargs):
            return vlm_bench.BenchmarkResult(
                prefill_sweep=vlm_bench.SweepData(x_values=[128, 256], tps_values=[10.0, 20.0], time_values=[0.1, 0.2]),
                decode_sweep=vlm_bench.SweepData(
                    x_values=[128, 256, 512],
                    tps_values=[30.0, 40.0, 50.0],
                    time_values=[0.3, 0.4, 0.5],
                ),
            )

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: object())
    monkeypatch.setattr(tps_cli, "_resolve_cli_batch_size", lambda args, pipeline: 2)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: _FakeTracker())
    monkeypatch.setattr(tps_cli, "_build_phase_trackers", lambda args, pipeline: (_FakeTracker(), _FakeTracker()))
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(tps_cli, "_stop_tracker_safe", lambda tracker: None)
    monkeypatch.setattr(tps_cli, "_extract_device_metric", lambda tracker: {"avg_power_w": 10.0})
    monkeypatch.setattr(
        tps_cli,
        "_extract_device_time_series",
        lambda tracker: {"power_w": [{"timestamp_s": 0.0, "value": 10.0}, {"timestamp_s": 1.0, "value": 10.0}]},
    )
    monkeypatch.setattr(
        "mblt_model_zoo.hf_transformers.utils.benchmark_utils.VLMTPSMeasurer",
        _FakeVLMTPSMeasurer,
    )

    assert tps_cli._run_vlm_sweep(args) == 0

    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    llm_run = payload["llm_results"]["runs"][0]
    llm_summary = payload["llm_results"]["summary"]
    vision_summary = payload["vision_results"][0]["summary"]
    rows = list(csv.DictReader(Path(args.csv).open(encoding="utf-8")))
    llm_rows = [row for row in rows if row["type"] == "llm"]
    prefill_tps_per_w = ((128 + 256) * 2) / 10.0
    decode_tps_per_w = (32 * 3 * 2) / 10.0

    assert vision_summary["vision_energy"]["mean"] == pytest.approx(10.0)
    assert llm_run["llm_prefill_energy"] == pytest.approx(10.0)
    assert llm_run["llm_decode_energy"] == pytest.approx(10.0)
    assert llm_run["llm_total_energy"] == pytest.approx(20.0)
    assert llm_summary["llm_prefill_energy"]["mean"] == pytest.approx(10.0)
    assert llm_summary["llm_decode_energy"]["mean"] == pytest.approx(10.0)
    assert llm_summary["llm_total_energy"]["mean"] == pytest.approx(20.0)
    assert llm_run["llm_prefill_tps_per_w"] == pytest.approx(prefill_tps_per_w)
    assert llm_run["llm_decode_tps_per_w"] == pytest.approx(decode_tps_per_w)
    assert llm_run["llm_prefill_j_per_tok"] == pytest.approx(10.0 / ((128 + 256) * 2))
    assert llm_run["llm_decode_j_per_tok"] == pytest.approx(10.0 / (32 * 3 * 2))
    assert llm_summary["llm_prefill_tps_per_w"]["mean"] == pytest.approx(prefill_tps_per_w)
    assert llm_summary["llm_decode_tps_per_w"]["mean"] == pytest.approx(decode_tps_per_w)
    assert [float(row["llm_prefill_tps_per_w"]) for row in llm_rows] == [pytest.approx(prefill_tps_per_w)]
    assert [float(row["llm_decode_tps_per_w"]) for row in llm_rows] == [pytest.approx(decode_tps_per_w)]
    assert [float(row["vision_energy_j"]) for row in llm_rows] == [pytest.approx(10.0)]
    assert [float(row["llm_prefill_energy_j"]) for row in llm_rows] == [pytest.approx(10.0)]
    assert [float(row["llm_decode_energy_j"]) for row in llm_rows] == [pytest.approx(10.0)]
    assert [float(row["llm_total_energy_j"]) for row in llm_rows] == [pytest.approx(20.0)]
    assert [float(row["total_energy_j"]) for row in llm_rows] == [pytest.approx(30.0)]


def test_text_sweep_token_helpers_use_whole_sweep_scope() -> None:
    """Verify text sweep token helpers match whole-sweep trace energy scope."""
    result = text_bench.BenchmarkResult(
        prefill_sweep=text_bench.SweepData(x_values=[128, 256], tps_values=[1.0, 2.0], time_values=[1.0, 1.0]),
        decode_sweep=text_bench.SweepData(
            x_values=[128, 256, 512],
            tps_values=[3.0, 4.0, 5.0],
            time_values=[1.0, 1.0, 1.0],
        ),
    )

    assert text_bench._sweep_prefill_token_count(result, batch_size=2) == (128 + 256) * 2
    assert text_bench._sweep_decode_token_count(result, decode_window=32, batch_size=2) == 32 * 3 * 2


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


@pytest.mark.parametrize(("backend", "expected"), [("npu", 1.0), ("gpu", 1.0), ("cpu", 1.0)])
def test_benchmark_common_tracker_interval_policy(backend: str, expected: float) -> None:
    """Verify tracker sampling intervals are fixed across resolved backends."""

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


def test_text_original_models_mixed_run_keeps_mobilint_npu() -> None:
    """Verify --original-models mixed runs route Mobilint rows to NPU/CPU, parents to CUDA/GPU."""
    argv = ["measure", "--original-models", "--model", "mobilint/model-a"]
    args = text_bench._build_arg_parser().parse_args(argv)
    text_bench._resolve_runtime_defaults(args, argv)

    mobilint_args = text_bench._args_for_target_device_backend(args, model_id="mobilint/model-a")
    parent_args = text_bench._args_for_target_device_backend(args, model_id="meta-llama/model-a")

    assert mobilint_args.device == "cpu"
    assert mobilint_args.device_backend == "npu"
    assert parent_args.device == "cuda"
    assert parent_args.device_backend == "gpu"


def test_text_original_models_mixed_run_preserves_explicit_device() -> None:
    """Verify explicit --device on Mobilint targets wins over the mixed-run override."""
    argv = ["measure", "--original-models", "--device", "cuda:0", "--model", "mobilint/model-a"]
    args = text_bench._build_arg_parser().parse_args(argv)
    text_bench._resolve_runtime_defaults(args, argv)

    mobilint_args = text_bench._args_for_target_device_backend(args, model_id="mobilint/model-a")

    assert mobilint_args.device == "cuda:0"


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


def test_text_measure_device_payload_requires_complete_energy_repeats() -> None:
    """Verify measure aggregate energy is omitted when any repeat lacks trace-integrated energy."""

    payload = text_bench._measure_device_payload(
        [
            {"avg_power_w": 4.0, "p99_power_w": 5.0, "total_energy_j": 2.0, "prefill_tps": 10.0},
            {"avg_power_w": 6.0, "p99_power_w": 7.0, "total_energy_j": None, "prefill_tps": 20.0},
        ]
    )

    assert payload is not None
    assert payload["avg_power_w"] == pytest.approx(5.0)
    assert payload["p99_power_w"] == pytest.approx(7.0)
    assert payload["total_energy_j"] is None
    assert payload["prefill_tps_last"] == pytest.approx(20.0)


def test_text_measure_device_payload_sums_complete_energy_repeats() -> None:
    """Verify measure aggregate energy is summed only when all repeats have energy."""

    payload = text_bench._measure_device_payload(
        [
            {"avg_power_w": 4.0, "total_energy_j": 2.0},
            {"avg_power_w": 6.0, "total_energy_j": 3.0},
        ]
    )

    assert payload is not None
    assert payload["total_energy_j"] == pytest.approx(5.0)


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


def test_text_skipped_sidecar_missing_returns_empty(tmp_path) -> None:
    """Verify sidecar reader treats missing files as an empty list."""
    assert text_bench._read_skipped_sidecar(tmp_path, "measure") == []
    assert text_bench._read_skipped_sidecar(tmp_path, "sweep") == []


def test_text_skipped_sidecar_filename_rejects_unknown_mode() -> None:
    """Verify the sidecar filename helper rejects modes other than measure/sweep."""
    with pytest.raises(ValueError):
        text_bench._skipped_sidecar_filename("bogus")


def test_text_skipped_sidecar_roundtrip(tmp_path) -> None:
    """Verify sidecar writer/reader roundtrips skip records per mode."""
    records = [
        {
            "model": "model-a",
            "device": "cuda:0",
            "batch_size": 8,
            "phase": "load",
            "skipped_reason": "cuda_oom",
        },
        {
            "model": "model-b",
            "device": "npu",
            "batch_size": 4,
            "phase": "measure",
            "skipped_reason": "npu_alloc",
            "npu_max_batch_size": 16,
        },
    ]
    text_bench._write_skipped_sidecar(tmp_path, records, "measure")
    assert (tmp_path / "skipped_records_measure.json").is_file()
    assert text_bench._read_skipped_sidecar(tmp_path, "measure") == records


def test_text_skipped_sidecar_modes_are_isolated(tmp_path) -> None:
    """Verify measure and sweep sidecars persist to independent files."""
    measure_records = [
        {
            "model": "measure-a",
            "batch_size": 4,
            "phase": "load",
            "skipped_reason": "cuda_oom",
        }
    ]
    sweep_records = [
        {
            "model": "sweep-a",
            "batch_size": 8,
            "phase": "measure",
            "skipped_reason": "npu_alloc",
        }
    ]

    text_bench._write_skipped_sidecar(tmp_path, measure_records, "measure")
    text_bench._write_skipped_sidecar(tmp_path, sweep_records, "sweep")

    assert (tmp_path / "skipped_records_measure.json").is_file()
    assert (tmp_path / "skipped_records_sweep.json").is_file()
    assert text_bench._read_skipped_sidecar(tmp_path, "measure") == measure_records
    assert text_bench._read_skipped_sidecar(tmp_path, "sweep") == sweep_records


def test_text_measure_rebuild_loads_skipped_sidecar(tmp_path) -> None:
    """Verify measure rebuild without explicit records reloads sidecar rows into CSV."""
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
    text_bench._write_skipped_sidecar(
        tmp_path,
        [
            {
                "model": "model-b",
                "batch_size": 32,
                "phase": "load",
                "skipped_reason": "cuda_oom",
            }
        ],
        "measure",
    )

    text_bench._rebuild_measure_outputs(tmp_path)

    csv_path = tmp_path / "combined_measure.csv"
    assert csv_path.is_file()
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    skipped = [row for row in rows if row.get("skipped_reason") == "cuda_oom"]
    assert len(skipped) == 1
    assert skipped[0]["model"] == "model-b"


def test_text_measure_rebuild_only_reads_measure_sidecar(tmp_path) -> None:
    """Verify measure rebuild ignores the sweep sidecar."""
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
    text_bench._write_skipped_sidecar(
        tmp_path,
        [
            {
                "model": "sweep-only",
                "batch_size": 8,
                "phase": "measure",
                "skipped_reason": "npu_alloc",
            }
        ],
        "sweep",
    )

    text_bench._rebuild_measure_outputs(tmp_path)

    rows = list(csv.DictReader((tmp_path / "combined_measure.csv").open("r", encoding="utf-8")))
    assert not [row for row in rows if row.get("model") == "sweep-only"]


def test_text_measure_rebuild_missing_sidecar_is_backward_compat(tmp_path) -> None:
    """Verify measure rebuild does not crash when the sidecar is absent."""
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


def test_text_sweep_rebuild_loads_skipped_sidecar(tmp_path) -> None:
    """Verify sweep rebuild without explicit records reloads sidecar rows into combined CSV."""
    payload = {
        "model": "sweep-a",
        "benchmark": {
            "prefill_sweep": {"x_values": [8], "tps_values": [10.0], "time_values": [0.8]},
            "decode_sweep": {"x_values": [4], "tps_values": [20.0], "time_values": [0.2]},
        },
    }
    (tmp_path / "sweep-a.json").write_text(json.dumps(payload), encoding="utf-8")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [
            {
                "model": "sweep-b",
                "batch_size": 16,
                "phase": "measure",
                "skipped_reason": "npu_alloc",
            }
        ],
        "sweep",
    )

    text_bench._rebuild_combined_outputs(tmp_path)

    csv_path = tmp_path / "combined.csv"
    assert csv_path.is_file()
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    skipped = [row for row in rows if row.get("skipped_reason") == "npu_alloc"]
    assert len(skipped) == 1
    assert skipped[0]["model"] == "sweep-b"


def test_text_sweep_rebuild_only_reads_sweep_sidecar(tmp_path) -> None:
    """Verify sweep rebuild ignores the measure sidecar."""
    payload = {
        "model": "sweep-a",
        "benchmark": {
            "prefill_sweep": {"x_values": [8], "tps_values": [10.0], "time_values": [0.8]},
            "decode_sweep": {"x_values": [4], "tps_values": [20.0], "time_values": [0.2]},
        },
    }
    (tmp_path / "sweep-a.json").write_text(json.dumps(payload), encoding="utf-8")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [
            {
                "model": "measure-only",
                "batch_size": 32,
                "phase": "load",
                "skipped_reason": "cuda_oom",
            }
        ],
        "measure",
    )

    text_bench._rebuild_combined_outputs(tmp_path)

    rows = list(csv.DictReader((tmp_path / "combined.csv").open("r", encoding="utf-8")))
    assert not [row for row in rows if row.get("model") == "measure-only"]


def test_text_sweep_rebuild_ignores_sidecar_glob(tmp_path) -> None:
    """Verify sweep rebuild does not treat the JSON sidecar as a benchmark payload."""
    text_bench._write_skipped_sidecar(
        tmp_path,
        [
            {
                "model": "sweep-b",
                "batch_size": 16,
                "phase": "measure",
                "skipped_reason": "npu_alloc",
            }
        ],
        "sweep",
    )

    text_bench._rebuild_combined_outputs(tmp_path)

    csv_path = tmp_path / "combined.csv"
    assert csv_path.is_file()
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert [row for row in rows if row.get("skipped_reason") == "npu_alloc"]


def test_text_rebuild_explicit_records_persist_sidecar(tmp_path) -> None:
    """Verify passing an explicit skipped_records list writes the mode-specific sidecar."""
    records = [
        {
            "model": "run-a",
            "batch_size": 8,
            "phase": "load",
            "skipped_reason": "cuda_oom",
        }
    ]

    text_bench._rebuild_measure_outputs(tmp_path, skipped_records=records)

    assert text_bench._read_skipped_sidecar(tmp_path, "measure") == records
    assert text_bench._read_skipped_sidecar(tmp_path, "sweep") == []


def _write_measure_result_json_for_reconcile(path: Path, label: str) -> None:
    """Minimal measure per-target JSON with the ``benchmark_type`` marker the reconciler expects."""
    path.write_text(
        json.dumps(
            {
                "model": label,
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
        ),
        encoding="utf-8",
    )


def _write_sweep_result_json_for_reconcile(path: Path, label: str) -> None:
    """Minimal sweep per-target JSON that :func:`reconcile_sidecar_and_disk` accepts as success."""
    path.write_text(
        json.dumps(
            {
                "model": label,
                "benchmark": {
                    "prefill_sweep": {"x_values": [8], "tps_values": [10.0], "time_values": [0.8]},
                    "decode_sweep": {"x_values": [4], "tps_values": [20.0], "time_values": [0.2]},
                },
            }
        ),
        encoding="utf-8",
    )


def _labels(records: list[dict[str, Any]]) -> list[str]:
    return [record.get("model") for record in records]


def test_reconcile_sidecar_only_skip_no_disk_payload_measure(tmp_path) -> None:
    """Task ``a7672`` baseline: only sidecar has X, no disk JSON — X is a skip."""
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-x", batch_size=8, phase="load", skipped_reason="cuda_oom")],
        "measure",
    )

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "measure")

    assert _labels(skips) == ["model-x"]
    assert successes == []


def test_reconcile_preloaded_sidecar_after_process_crash_measure(tmp_path) -> None:
    """Task ``1e113`` preload: a prior process's sidecar is preserved when nothing on disk conflicts."""
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-x", batch_size=8, phase="load", skipped_reason="cuda_oom")],
        "measure",
    )

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "measure")

    assert _labels(skips) == ["model-x"]
    assert successes == []


def test_reconcile_retry_replace_keeps_newer_detail_measure(tmp_path) -> None:
    """Task ``b2199`` retry-dedup interaction: latest ``_replace_skip_record`` row wins as a skip."""
    skipped_records: list[dict[str, Any]] = []
    text_bench._replace_skip_record(
        skipped_records,
        _sidecar_row_now(model="model-x", batch_size=64, phase="load", skipped_reason="cuda_oom", detail="first"),
    )
    text_bench._replace_skip_record(
        skipped_records,
        _sidecar_row_now(model="model-x", batch_size=32, phase="measure", skipped_reason="npu_alloc", detail="second"),
    )
    text_bench._write_skipped_sidecar(tmp_path, skipped_records, "measure")

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "measure")

    assert _labels(skips) == ["model-x"]
    assert skips[0]["detail"] == "second"
    assert successes == []


def test_reconcile_disk_newer_drops_stale_sidecar_measure(tmp_path) -> None:
    """Task ``81f7a`` orphan-sidecar: an older sidecar row is dropped when the disk JSON is newer."""
    _write_measure_result_json_for_reconcile(tmp_path / "model-x_measure.json", "model-x")
    stale_row = {"model": "model-x", "phase": "load", "skipped_reason": "cuda_oom", "recorded_at": 0.0}
    text_bench._write_skipped_sidecar(tmp_path, [stale_row], "measure")

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "measure")

    assert skips == []
    assert [payload.get("model") for payload in successes] == ["model-x"]


def test_reconcile_sidecar_newer_supersedes_stale_disk_measure(tmp_path) -> None:
    """Task ``2efaa`` fresh-failure-over-stale-success: newer sidecar row wins; JSON stays on disk."""
    _write_measure_result_json_for_reconcile(tmp_path / "model-x_measure.json", "model-x")
    _backdate(tmp_path / "model-x_measure.json")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-x", batch_size=64, phase="load", skipped_reason="cuda_oom")],
        "measure",
    )

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "measure")

    assert _labels(skips) == ["model-x"]
    assert successes == []
    assert (tmp_path / "model-x_measure.json").exists()


def test_reconcile_rebuild_only_pass_same_result_as_normal_run_measure(tmp_path) -> None:
    """Task ``ea993`` rebuild-only: reconciler result is identical to the normal-run path."""
    _write_measure_result_json_for_reconcile(tmp_path / "model-x_measure.json", "model-x")
    _backdate(tmp_path / "model-x_measure.json")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-x", batch_size=64, phase="load", skipped_reason="cuda_oom")],
        "measure",
    )

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "measure")

    assert _labels(skips) == ["model-x"]
    assert successes == []


def test_reconcile_rebuild_charts_preserves_newer_sidecar_measure(tmp_path) -> None:
    """Task ``679ef`` rebuild-charts preservation: a newer sidecar row survives standalone rebuild."""
    _write_measure_result_json_for_reconcile(tmp_path / "model-x_measure.json", "model-x")
    _backdate(tmp_path / "model-x_measure.json")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-x", batch_size=64, phase="load", skipped_reason="cuda_oom")],
        "measure",
    )

    text_bench._rebuild_measure_outputs(tmp_path)

    persisted = text_bench._read_skipped_sidecar(tmp_path, "measure")
    assert _labels(persisted) == ["model-x"]


def test_reconcile_todays_case_stale_preloaded_sidecar_new_success_wins(tmp_path) -> None:
    """Discussion ``r3804234997``: older preloaded sidecar row + newer retry-success JSON → success wins."""
    stale_row = {
        "model": "model-x",
        "batch_size": 8,
        "phase": "load",
        "skipped_reason": "cuda_oom",
        "recorded_at": time.time() - _SKIP_TIMESTAMP_GAP_S,
    }
    text_bench._write_skipped_sidecar(tmp_path, [stale_row], "measure")
    _write_measure_result_json_for_reconcile(tmp_path / "model-x_measure.json", "model-x")

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "measure")

    assert skips == []
    assert [payload.get("model") for payload in successes] == ["model-x"]


def test_reconcile_legacy_row_without_recorded_at_defers_to_disk_measure(tmp_path) -> None:
    """Legacy rows written before the timestamp refactor default to epoch 0; disk JSON wins."""
    _write_measure_result_json_for_reconcile(tmp_path / "model-x_measure.json", "model-x")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [{"model": "model-x", "phase": "load", "skipped_reason": "cuda_oom"}],
        "measure",
    )

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "measure")

    assert skips == []
    assert [payload.get("model") for payload in successes] == ["model-x"]


def test_reconcile_sweep_sidecar_only_skip_no_disk_payload(tmp_path) -> None:
    """Sweep sibling of ``a7672`` baseline: sidecar-only entry is a skip."""
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="sweep-x", batch_size=16, phase="measure", skipped_reason="npu_alloc")],
        "sweep",
    )

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "sweep")

    assert _labels(skips) == ["sweep-x"]
    assert successes == []


def test_reconcile_sweep_disk_newer_drops_stale_sidecar(tmp_path) -> None:
    """Sweep sibling of ``81f7a``: newer sweep JSON drops the older sidecar row."""
    _write_sweep_result_json_for_reconcile(tmp_path / "sweep-x.json", "sweep-x")
    stale_row = {"model": "sweep-x", "phase": "measure", "skipped_reason": "npu_alloc", "recorded_at": 0.0}
    text_bench._write_skipped_sidecar(tmp_path, [stale_row], "sweep")

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "sweep")

    assert skips == []
    assert [payload.get("model") for payload in successes] == ["sweep-x"]


def test_reconcile_sweep_sidecar_newer_supersedes_stale_disk(tmp_path) -> None:
    """Sweep sibling of ``2efaa``: newer sidecar row wins over the backdated sweep JSON."""
    _write_sweep_result_json_for_reconcile(tmp_path / "sweep-x.json", "sweep-x")
    _backdate(tmp_path / "sweep-x.json")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="sweep-x", batch_size=16, phase="measure", skipped_reason="npu_alloc")],
        "sweep",
    )

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "sweep")

    assert _labels(skips) == ["sweep-x"]
    assert successes == []


def test_reconcile_ignores_other_mode_disk_payloads(tmp_path) -> None:
    """Sweep JSON must not participate in the measure reconciliation (and vice versa)."""
    _write_sweep_result_json_for_reconcile(tmp_path / "model-x.json", "model-x")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-x", phase="load", skipped_reason="cuda_oom")],
        "measure",
    )

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "measure")

    assert _labels(skips) == ["model-x"]
    assert successes == []


def test_reconcile_uses_caller_provided_sidecar_rows_measure(tmp_path) -> None:
    """``sidecar_rows`` overrides the on-disk sidecar for the current reconcile call."""
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-persisted", phase="load", skipped_reason="cuda_oom")],
        "measure",
    )
    override_rows = [_sidecar_row_now(model="model-override", phase="measure", skipped_reason="npu_alloc")]

    skips, successes = text_bench.reconcile_sidecar_and_disk(tmp_path, "measure", sidecar_rows=override_rows)

    assert _labels(skips) == ["model-override"]
    assert successes == []


def test_reconcile_rejects_unknown_benchmark_type(tmp_path) -> None:
    """The reconciler forwards the sidecar filename validation for unknown modes."""
    with pytest.raises(ValueError):
        text_bench.reconcile_sidecar_and_disk(tmp_path, "bogus")


def test_text_replace_skip_record_dedups_same_target_retry() -> None:
    """Verify a retried failure for the same target replaces the earlier row instead of duplicating it."""
    skipped_records: list[dict[str, Any]] = []

    text_bench._replace_skip_record(
        skipped_records,
        {"model": "model-a", "batch_size": 64, "phase": "load", "skipped_reason": "cuda_oom"},
    )
    text_bench._replace_skip_record(
        skipped_records,
        {"model": "model-a", "batch_size": 64, "phase": "load", "skipped_reason": "cuda_oom"},
    )

    assert skipped_records == [
        {"model": "model-a", "batch_size": 64, "phase": "load", "skipped_reason": "cuda_oom"},
    ]


def test_text_replace_skip_record_later_attempt_wins() -> None:
    """Verify the LATER failure record wins when a target retries at a different batch size."""
    skipped_records: list[dict[str, Any]] = []

    text_bench._replace_skip_record(
        skipped_records,
        {"model": "model-a", "batch_size": 64, "phase": "load", "skipped_reason": "cuda_oom"},
    )
    text_bench._replace_skip_record(
        skipped_records,
        {"model": "model-a", "batch_size": 32, "phase": "measure", "skipped_reason": "npu_alloc"},
    )

    assert skipped_records == [
        {"model": "model-a", "batch_size": 32, "phase": "measure", "skipped_reason": "npu_alloc"},
    ]


def test_text_replace_skip_record_preserves_other_targets() -> None:
    """Verify replacing one target's row does not disturb rows for other targets."""
    skipped_records: list[dict[str, Any]] = [
        {"model": "model-a", "batch_size": 64, "phase": "load", "skipped_reason": "cuda_oom"},
        {"model": "model-b", "batch_size": 8, "phase": "measure", "skipped_reason": "npu_alloc"},
    ]

    text_bench._replace_skip_record(
        skipped_records,
        {"model": "model-a", "batch_size": 32, "phase": "load", "skipped_reason": "cuda_oom"},
    )

    assert skipped_records == [
        {"model": "model-b", "batch_size": 8, "phase": "measure", "skipped_reason": "npu_alloc"},
        {"model": "model-a", "batch_size": 32, "phase": "load", "skipped_reason": "cuda_oom"},
    ]


def _write_measure_result_json(path: Path, label: str) -> None:
    """Write a minimal measure per-target JSON payload for reconciliation tests."""
    payload = {
        "model": label,
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
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_sweep_result_json(path: Path, label: str) -> None:
    """Write a minimal sweep per-target JSON payload for reconciliation tests."""
    payload = {
        "model": label,
        "benchmark": {
            "prefill_sweep": {"x_values": [8], "tps_values": [10.0], "time_values": [0.8]},
            "decode_sweep": {"x_values": [4], "tps_values": [20.0], "time_values": [0.2]},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_text_rebuild_measure_charts_preserves_sidecar_fresh_failure(tmp_path) -> None:
    """``--rebuild-charts`` alone must not sweep preserved fresh failures out of the sidecar.

    Scenario: an earlier run wrote ``X_measure.json`` at one ``--batch-size``,
    a later run re-ran X at a bigger ``--batch-size`` and it OOMed. Task
    ``2efaa`` preserved the fresh failure in ``skipped_records_measure.json``
    and task ``ea993`` masked the stale success from that run's combined
    output. A subsequent standalone ``--rebuild-charts`` must keep treating
    the preloaded skip row as authoritative because its ``recorded_at`` is
    strictly newer than the ``X_measure.json`` file's mtime.
    """
    _write_measure_result_json(tmp_path / "model-x_measure.json", "model-x")
    _backdate(tmp_path / "model-x_measure.json")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-x", batch_size=64, phase="load", skipped_reason="cuda_oom")],
        "measure",
    )

    text_bench._rebuild_measure_outputs(tmp_path)

    persisted = text_bench._read_skipped_sidecar(tmp_path, "measure")
    assert [record["model"] for record in persisted] == ["model-x"]
    assert persisted[0]["skipped_reason"] == "cuda_oom"
    rows = list(csv.DictReader((tmp_path / "combined_measure.csv").open("r", encoding="utf-8")))
    passing_rows = [row for row in rows if row.get("model") == "model-x" and row.get("skipped_reason", "") == ""]
    assert passing_rows == []
    skipped_rows = [row for row in rows if row.get("model") == "model-x" and row.get("skipped_reason") == "cuda_oom"]
    assert len(skipped_rows) == 1


def test_text_rebuild_measure_charts_keeps_orphan_skip_row(tmp_path) -> None:
    """Regression guard: an orphan skip (no on-disk JSON) is retained after ``--rebuild-charts``.

    Preload the sidecar with a skip row for ``model-x`` while only an
    unrelated ``model-y_measure.json`` exists on disk. The reconciler must
    leave ``model-x``'s row untouched — the disk-union only contains
    ``model-y`` and does not intersect the sidecar.
    """
    _write_measure_result_json(tmp_path / "model-y_measure.json", "model-y")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [{"model": "model-x", "batch_size": 32, "phase": "measure", "skipped_reason": "npu_alloc"}],
        "measure",
    )

    text_bench._rebuild_measure_outputs(tmp_path)

    persisted = text_bench._read_skipped_sidecar(tmp_path, "measure")
    assert [record["model"] for record in persisted] == ["model-x"]
    rows = list(csv.DictReader((tmp_path / "combined_measure.csv").open("r", encoding="utf-8")))
    y_rows = [row for row in rows if row.get("model") == "model-y" and row.get("skipped_reason", "") == ""]
    assert len(y_rows) == 1
    x_skipped = [row for row in rows if row.get("model") == "model-x" and row.get("skipped_reason") == "npu_alloc"]
    assert len(x_skipped) == 1


def test_text_rebuild_measure_charts_preserve_fresh_failure_is_idempotent(tmp_path) -> None:
    """Repeated ``--rebuild-charts`` invocations must not drift the sidecar or CSV.

    Same setup as ``test_text_rebuild_measure_charts_preserves_sidecar_fresh_failure``:
    call ``_rebuild_measure_outputs`` twice back-to-back and assert the
    persisted sidecar and combined CSV are byte-for-byte unchanged between
    invocations.
    """
    _write_measure_result_json(tmp_path / "model-x_measure.json", "model-x")
    _backdate(tmp_path / "model-x_measure.json")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-x", batch_size=64, phase="load", skipped_reason="cuda_oom")],
        "measure",
    )

    text_bench._rebuild_measure_outputs(tmp_path)
    sidecar_after_first = (tmp_path / "skipped_records_measure.json").read_text(encoding="utf-8")
    csv_after_first = (tmp_path / "combined_measure.csv").read_text(encoding="utf-8")

    text_bench._rebuild_measure_outputs(tmp_path)
    sidecar_after_second = (tmp_path / "skipped_records_measure.json").read_text(encoding="utf-8")
    csv_after_second = (tmp_path / "combined_measure.csv").read_text(encoding="utf-8")

    assert sidecar_after_first == sidecar_after_second
    assert csv_after_first == csv_after_second


def test_text_rebuild_sweep_charts_preserves_sidecar_fresh_failure(tmp_path) -> None:
    """Sweep sibling of the preserved-fresh-failure rebuild guard.

    Mirrors ``test_text_rebuild_measure_charts_preserves_sidecar_fresh_failure``
    against ``_rebuild_combined_outputs`` so the sweep code path receives the
    same protection: sidecar row's ``recorded_at`` beats the backdated on-disk
    payload's mtime.
    """
    _write_sweep_result_json(tmp_path / "sweep-x.json", "sweep-x")
    _backdate(tmp_path / "sweep-x.json")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="sweep-x", batch_size=16, phase="measure", skipped_reason="npu_alloc")],
        "sweep",
    )

    text_bench._rebuild_combined_outputs(tmp_path)

    persisted = text_bench._read_skipped_sidecar(tmp_path, "sweep")
    assert [record["model"] for record in persisted] == ["sweep-x"]
    assert persisted[0]["skipped_reason"] == "npu_alloc"
    rows = list(csv.DictReader((tmp_path / "combined.csv").open("r", encoding="utf-8")))
    passing_rows = [row for row in rows if row.get("model") == "sweep-x" and row.get("skipped_reason", "") == ""]
    assert passing_rows == []
    skipped_rows = [row for row in rows if row.get("model") == "sweep-x" and row.get("skipped_reason") == "npu_alloc"]
    assert len(skipped_rows) == 1


def test_text_rebuild_sweep_charts_keeps_orphan_skip_row(tmp_path) -> None:
    """Sweep sibling of the orphan-skip regression guard."""
    _write_sweep_result_json(tmp_path / "sweep-y.json", "sweep-y")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [{"model": "sweep-x", "batch_size": 32, "phase": "measure", "skipped_reason": "npu_alloc"}],
        "sweep",
    )

    text_bench._rebuild_combined_outputs(tmp_path)

    persisted = text_bench._read_skipped_sidecar(tmp_path, "sweep")
    assert [record["model"] for record in persisted] == ["sweep-x"]
    rows = list(csv.DictReader((tmp_path / "combined.csv").open("r", encoding="utf-8")))
    y_rows = [row for row in rows if row.get("model") == "sweep-y" and row.get("skipped_reason", "") == ""]
    assert y_rows, "unrelated sweep success payload must still contribute rows"
    x_skipped = [row for row in rows if row.get("model") == "sweep-x" and row.get("skipped_reason") == "npu_alloc"]
    assert len(x_skipped) == 1


def test_text_rebuild_sweep_charts_preserve_fresh_failure_is_idempotent(tmp_path) -> None:
    """Sweep sibling of the preserved-fresh-failure idempotency guard."""
    _write_sweep_result_json(tmp_path / "sweep-x.json", "sweep-x")
    _backdate(tmp_path / "sweep-x.json")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="sweep-x", batch_size=16, phase="measure", skipped_reason="npu_alloc")],
        "sweep",
    )

    text_bench._rebuild_combined_outputs(tmp_path)
    sidecar_after_first = (tmp_path / "skipped_records_sweep.json").read_text(encoding="utf-8")
    csv_after_first = (tmp_path / "combined.csv").read_text(encoding="utf-8")

    text_bench._rebuild_combined_outputs(tmp_path)
    sidecar_after_second = (tmp_path / "skipped_records_sweep.json").read_text(encoding="utf-8")
    csv_after_second = (tmp_path / "combined.csv").read_text(encoding="utf-8")

    assert sidecar_after_first == sidecar_after_second
    assert csv_after_first == csv_after_second


def test_text_rebuild_measure_excludes_stale_success_for_skipped_target(tmp_path) -> None:
    """Fresh failure passed via ``skipped_records=`` excludes stale on-disk success.

    Follow-up to task ``2efaa``: when a fresh in-process failure for ``model-x``
    is passed through ``skipped_records`` while a prior ``model-x_measure.json``
    still sits on disk, the rebuilt combined CSV must contain only the fresh
    skip row — not a duplicate passing row derived from the stale JSON. The
    fresh row carries ``recorded_at = time.time()`` while the on-disk JSON is
    backdated, so timestamp precedence keeps the skip.
    """
    _write_measure_result_json(tmp_path / "model-x_measure.json", "model-x")
    _backdate(tmp_path / "model-x_measure.json")
    skipped_records: list[dict[str, Any]] = [
        _sidecar_row_now(model="model-x", batch_size=64, phase="load", skipped_reason="cuda_oom"),
    ]

    text_bench._rebuild_measure_outputs(tmp_path, skipped_records=skipped_records)

    # Stale JSON stays on disk for manual inspection.
    assert (tmp_path / "model-x_measure.json").exists()
    rows = list(csv.DictReader((tmp_path / "combined_measure.csv").open("r", encoding="utf-8")))
    passing_rows = [row for row in rows if row.get("model") == "model-x" and row.get("skipped_reason", "") == ""]
    assert passing_rows == []
    skipped_rows = [row for row in rows if row.get("model") == "model-x" and row.get("skipped_reason") == "cuda_oom"]
    assert len(skipped_rows) == 1


def test_text_rebuild_measure_keeps_success_when_sidecar_empty(tmp_path) -> None:
    """Regression guard: no matching sidecar row keeps the on-disk success payload."""
    _write_measure_result_json(tmp_path / "model-x_measure.json", "model-x")

    text_bench._rebuild_measure_outputs(tmp_path)

    rows = list(csv.DictReader((tmp_path / "combined_measure.csv").open("r", encoding="utf-8")))
    model_rows = [row for row in rows if row.get("model") == "model-x"]
    assert len(model_rows) == 1
    assert model_rows[0].get("skipped_reason", "") == ""


def test_text_rebuild_measure_keeps_success_for_unrelated_target(tmp_path) -> None:
    """Unrelated Y success + X skip yields both rows without cross-contamination."""
    _write_measure_result_json(tmp_path / "model-y_measure.json", "model-y")
    skipped_records: list[dict[str, Any]] = [
        {"model": "model-x", "batch_size": 32, "phase": "measure", "skipped_reason": "npu_alloc"},
    ]

    text_bench._rebuild_measure_outputs(tmp_path, skipped_records=skipped_records)

    rows = list(csv.DictReader((tmp_path / "combined_measure.csv").open("r", encoding="utf-8")))
    labels_and_reason = [(row.get("model"), row.get("skipped_reason", "")) for row in rows]
    assert ("model-y", "") in labels_and_reason
    assert ("model-x", "npu_alloc") in labels_and_reason
    assert ("model-x", "") not in labels_and_reason


def test_text_rebuild_sweep_excludes_stale_success_for_skipped_target(tmp_path) -> None:
    """Sweep sibling: fresh failure via ``skipped_records=`` excludes stale sweep JSON."""
    _write_sweep_result_json(tmp_path / "sweep-x.json", "sweep-x")
    _backdate(tmp_path / "sweep-x.json")
    skipped_records: list[dict[str, Any]] = [
        _sidecar_row_now(model="sweep-x", batch_size=16, phase="measure", skipped_reason="npu_alloc"),
    ]

    text_bench._rebuild_combined_outputs(tmp_path, skipped_records=skipped_records)

    assert (tmp_path / "sweep-x.json").exists()
    rows = list(csv.DictReader((tmp_path / "combined.csv").open("r", encoding="utf-8")))
    passing_rows = [row for row in rows if row.get("model") == "sweep-x" and row.get("skipped_reason", "") == ""]
    assert passing_rows == []
    skipped_rows = [row for row in rows if row.get("model") == "sweep-x" and row.get("skipped_reason") == "npu_alloc"]
    assert len(skipped_rows) == 1


def test_text_rebuild_sweep_keeps_success_for_unrelated_target(tmp_path) -> None:
    """Sweep sibling: unrelated Y success + X skip yields both rows."""
    _write_sweep_result_json(tmp_path / "sweep-y.json", "sweep-y")
    skipped_records: list[dict[str, Any]] = [
        {"model": "sweep-x", "batch_size": 16, "phase": "measure", "skipped_reason": "npu_alloc"},
    ]

    text_bench._rebuild_combined_outputs(tmp_path, skipped_records=skipped_records)

    rows = list(csv.DictReader((tmp_path / "combined.csv").open("r", encoding="utf-8")))
    labels_and_reason = {(row.get("model"), row.get("skipped_reason", "")) for row in rows}
    assert ("sweep-y", "") in labels_and_reason
    assert ("sweep-x", "npu_alloc") in labels_and_reason
    assert ("sweep-x", "") not in labels_and_reason


_MEASURE_CHART_FILENAMES = (
    "measure_prefill_tps.png",
    "measure_prefill_tps_per_w.png",
    "measure_decode_tps.png",
    "measure_decode_tps_per_w.png",
    "measure_avg_power_w.png",
    "measure_avg_temperature_c.png",
    "measure_avg_utilization_pct.png",
    "measure_avg_memory_used_mb.png",
    "measure_total_energy_j.png",
)


_SWEEP_CHART_FILENAMES = (
    "prefill_tps.png",
    "prefill_tps_per_w.png",
    "decode_tps.png",
    "decode_tps_per_w.png",
    "avg_power_w.png",
    "avg_temperature_c.png",
    "avg_utilization_pct.png",
    "avg_memory_used_mb.png",
    "total_energy_j.png",
)


def test_text_rebuild_measure_overwrites_stale_markdown_when_all_superseded(tmp_path) -> None:
    """Stale success markdown must be overwritten when every target is superseded by a newer skip.

    Follow-up to PR #109 review: after the timestamp reconciler returns
    ``skips=[X_skip]`` and ``successes=[]``, ``_write_measure_markdown`` must
    still overwrite the on-disk file so the stale success table left by a
    prior rebuild does not linger above the freshly appended skip section.
    """
    _write_measure_result_json(tmp_path / "model-x_measure.json", "model-x")
    _backdate(tmp_path / "model-x_measure.json")
    combined_md = tmp_path / "combined_measure.md"
    combined_md.write_text(
        "| model | prefill_tps_mean |\n| --- | ---: |\n| model-x | 10.0 |\n",
        encoding="utf-8",
    )
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-x", batch_size=64, phase="load", skipped_reason="cuda_oom")],
        "measure",
    )

    text_bench._rebuild_measure_outputs(tmp_path)

    content = combined_md.read_text(encoding="utf-8")
    assert "_No successful measure results._" in content
    assert "| model-x | 10.0 |" not in content
    assert "## Skipped Targets" in content
    assert "cuda_oom" in content


def test_text_rebuild_measure_removes_stale_measure_png_when_all_superseded(tmp_path) -> None:
    """Stale measure PNGs must be removed when every target is superseded by a newer skip."""
    _write_measure_result_json(tmp_path / "model-x_measure.json", "model-x")
    _backdate(tmp_path / "model-x_measure.json")
    for filename in _MEASURE_CHART_FILENAMES:
        (tmp_path / filename).write_bytes(b"stale-png")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="model-x", batch_size=64, phase="load", skipped_reason="cuda_oom")],
        "measure",
    )

    text_bench._rebuild_measure_outputs(tmp_path)

    for filename in _MEASURE_CHART_FILENAMES:
        assert not (tmp_path / filename).exists(), f"stale PNG {filename} should be removed"


def test_text_rebuild_measure_writes_success_markdown_when_only_success(tmp_path) -> None:
    """Regression: on a clean success-only rebuild, the markdown carries the real table."""
    _write_measure_result_json(tmp_path / "model-x_measure.json", "model-x")

    text_bench._rebuild_measure_outputs(tmp_path)

    content = (tmp_path / "combined_measure.md").read_text(encoding="utf-8")
    assert "_No successful measure results._" not in content
    assert "model-x" in content
    assert "prefill_tps_mean" in content


def test_text_rebuild_sweep_overwrites_stale_markdown_when_all_superseded(tmp_path) -> None:
    """Sweep sibling of the measure stale-markdown overwrite guard."""
    _write_sweep_result_json(tmp_path / "sweep-x.json", "sweep-x")
    _backdate(tmp_path / "sweep-x.json")
    combined_md = tmp_path / "combined.md"
    combined_md.write_text(
        "| model | prefill_tps_8 |\n| --- | ---: |\n| sweep-x | 10.0 |\n",
        encoding="utf-8",
    )
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="sweep-x", batch_size=16, phase="measure", skipped_reason="npu_alloc")],
        "sweep",
    )

    text_bench._rebuild_combined_outputs(tmp_path)

    content = combined_md.read_text(encoding="utf-8")
    assert "_No successful sweep results._" in content
    assert "| sweep-x | 10.0 |" not in content
    assert "## Skipped Targets" in content
    assert "npu_alloc" in content


def test_text_rebuild_sweep_removes_stale_sweep_png_when_all_superseded(tmp_path) -> None:
    """Sweep sibling of the measure stale-PNG cleanup guard."""
    _write_sweep_result_json(tmp_path / "sweep-x.json", "sweep-x")
    _backdate(tmp_path / "sweep-x.json")
    for filename in _SWEEP_CHART_FILENAMES:
        (tmp_path / filename).write_bytes(b"stale-png")
    text_bench._write_skipped_sidecar(
        tmp_path,
        [_sidecar_row_now(model="sweep-x", batch_size=16, phase="measure", skipped_reason="npu_alloc")],
        "sweep",
    )

    text_bench._rebuild_combined_outputs(tmp_path)

    for filename in _SWEEP_CHART_FILENAMES:
        assert not (tmp_path / filename).exists(), f"stale PNG {filename} should be removed"


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


def test_text_benchmark_include_private_flag_defaults_false_and_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify --include-private threads through text-generation default target resolution."""
    observed: list[bool] = []

    def fake_list_default_model_ids(task, *, include_private=False):  # type: ignore[no-untyped-def]
        observed.append(bool(include_private))
        return ["mobilint/text-a"]

    monkeypatch.setattr(text_bench, "list_default_model_ids", fake_list_default_model_ids)

    default_args = text_bench._build_arg_parser().parse_args(["measure"])
    assert default_args.include_private is False
    text_bench._collect_text_run_targets(default_args)

    private_args = text_bench._build_arg_parser().parse_args(["measure", "--include-private"])
    assert private_args.include_private is True
    text_bench._collect_text_run_targets(private_args)

    assert observed == [False, True]


def test_vlm_benchmark_include_private_flag_defaults_false_and_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify --include-private threads through image-text-to-text default target resolution."""
    observed: list[bool] = []

    def fake_list_default_model_ids(task, *, include_private=False):  # type: ignore[no-untyped-def]
        observed.append(bool(include_private))
        return ["mobilint/vlm-a"]

    monkeypatch.setattr(vlm_bench, "list_default_model_ids", fake_list_default_model_ids)
    monkeypatch.setattr(vlm_bench, "_filter_text_targets_by_batch_mode", lambda targets, **_: [])

    default_args = vlm_bench._build_arg_parser().parse_args(["measure"])
    assert default_args.include_private is False
    vlm_bench._collect_vlm_run_targets(default_args)

    private_args = vlm_bench._build_arg_parser().parse_args(["measure", "--include-private"])
    assert private_args.include_private is True
    vlm_bench._collect_vlm_run_targets(private_args)

    assert observed == [False, True]


@pytest.mark.parametrize("command", ["measure", "sweep"])
def test_text_benchmark_parses_dev_no_and_batch_size(command: str) -> None:
    """Verify --dev-no and --batch-size parse on both text-generation subcommands."""
    args = text_bench._build_arg_parser().parse_args([command, "--dev-no", "0,1,2,3", "--batch-size", "64"])

    assert args.dev_no == [0, 1, 2, 3]
    assert args.batch_size == 64

    scalar = text_bench._build_arg_parser().parse_args([command, "--dev-no", "1"])
    assert scalar.dev_no == 1


def test_text_target_filtering_admits_original_under_batch_size_override(monkeypatch) -> None:
    """Verify --original-models with cfg max_batch_size=1 is admitted under --batch --batch-size N>1."""
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = [
        ("upstream/original", [None], "original", "original", None),
    ]

    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: False)
    monkeypatch.setattr(text_bench, "_resolve_config_max_batch_size", lambda model_id, revision, *, task: 1)

    admitted = text_bench._filter_text_targets_by_batch_mode(
        raw_targets,
        batch_mode="batch",
        override_batch_size=32,
        original_models=True,
    )

    assert len(admitted) == 1
    assert admitted[0].model_id == "upstream/original"
    assert admitted[0].batch_mode == "batch"
    assert admitted[0].max_batch_size == 32


def test_text_target_filtering_rejects_non_batch_mobilint_under_batch_size_override(monkeypatch) -> None:
    """Verify non-batch Mobilint MXQs are filtered out of --batch --batch-size N>1 without --original-models.

    Regression guard: previously the ``forced_batch`` relaxation applied to every target under
    ``batch_mode=batch`` with ``batch_size>1``, causing config max_batch_size==1 Mobilint targets
    to fan out into ``N=batch_size`` slots and hit ``MobilintBackendAllocError``.
    """
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = [
        ("mobilint/non-batch", [None], "non-batch", "non-batch", None),
    ]

    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: False)
    monkeypatch.setattr(text_bench, "_resolve_config_max_batch_size", lambda model_id, revision, *, task: 1)

    batch = text_bench._filter_text_targets_by_batch_mode(
        raw_targets,
        batch_mode="batch",
        override_batch_size=64,
        original_models=False,
    )
    non_batch = text_bench._filter_text_targets_by_batch_mode(
        raw_targets,
        batch_mode="non_batch",
        override_batch_size=64,
        original_models=False,
    )

    assert batch == []
    assert [target.model_id for target in non_batch] == ["mobilint/non-batch"]


def test_text_target_filtering_batch_size_override_applies_to_batch_mobilint(monkeypatch) -> None:
    """Verify --batch-size overrides the effective batch dim on already-batch-eligible Mobilint targets."""
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = [
        ("mobilint/batch16", [None], "batch16", "batch16", None),
    ]

    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: False)
    monkeypatch.setattr(text_bench, "_resolve_config_max_batch_size", lambda model_id, revision, *, task: 16)

    admitted = text_bench._filter_text_targets_by_batch_mode(
        raw_targets,
        batch_mode="batch",
        override_batch_size=64,
        original_models=False,
    )

    assert len(admitted) == 1
    assert admitted[0].model_id == "mobilint/batch16"
    assert admitted[0].batch_mode == "batch"
    assert admitted[0].max_batch_size == 64


def test_text_target_filtering_original_models_mixed_admits_only_batch_and_upstream(monkeypatch) -> None:
    """Verify --original-models mixed batch admits batch mobilint + upstream, drops non-batch mobilint."""
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = [
        ("mobilint/non-batch", [None], "non-batch", "non-batch", None),
        ("mobilint/batch16", [None], "batch16", "batch16", None),
        ("upstream/original", [None], "original", "original", None),
    ]

    cfg_map = {
        "mobilint/non-batch": 1,
        "mobilint/batch16": 16,
        "upstream/original": 1,
    }
    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: False)
    monkeypatch.setattr(
        text_bench,
        "_resolve_config_max_batch_size",
        lambda model_id, revision, *, task: cfg_map[model_id],
    )

    admitted = text_bench._filter_text_targets_by_batch_mode(
        raw_targets,
        batch_mode="batch",
        override_batch_size=64,
        original_models=True,
    )

    assert sorted(target.model_id for target in admitted) == ["mobilint/batch16", "upstream/original"]
    for target in admitted:
        assert target.batch_mode == "batch"
        assert target.max_batch_size == 64


def _stub_text_target_hub_probes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_batch_size_map: dict[str, int] | None = None,
    available_model_ids: list[str] | None = None,
) -> None:
    """Stub Hub-facing helpers used by _collect_text_run_targets to keep the test hermetic."""
    placeholder_ids = list(available_model_ids) if available_model_ids else ["mobilint/placeholder"]
    monkeypatch.setattr(
        text_bench,
        "list_default_model_ids",
        lambda task, *, include_private=False: list(placeholder_ids),
    )
    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: False)
    monkeypatch.setattr(text_bench, "_is_gguf_model_id", lambda model_id: False)
    cfg_map = dict(max_batch_size_map or {})
    monkeypatch.setattr(
        text_bench,
        "_resolve_config_max_batch_size",
        lambda model_id, revision, *, task: cfg_map.get(model_id, 1),
    )


def test_collect_text_run_targets_original_models_preserves_mobilint(monkeypatch, tmp_path) -> None:
    """Verify --original-models + a caller-listed Mobilint id keeps BOTH Mobilint and its parent."""
    monkeypatch.setattr(
        text_bench,
        "_resolve_original_model_ids",
        lambda model_ids: ["meta-llama/Llama-3.1-8B-Instruct"],
    )
    _stub_text_target_hub_probes(
        monkeypatch,
        max_batch_size_map={
            "mobilint/Llama-3.1-8B-Instruct-Batch16": 16,
            "meta-llama/Llama-3.1-8B-Instruct": 1,
        },
    )

    args = text_bench._build_arg_parser().parse_args(
        [
            "measure",
            "--batch",
            "--original-models",
            "--model",
            "mobilint/Llama-3.1-8B-Instruct-Batch16",
            "--batch-size",
            "64",
            "--output-dir",
            str(tmp_path),
        ]
    )

    _, run_targets = text_bench._collect_text_run_targets(args)

    model_ids = [entry[0] for entry in run_targets]
    assert set(model_ids) == {
        "mobilint/Llama-3.1-8B-Instruct-Batch16",
        "meta-llama/Llama-3.1-8B-Instruct",
    }
    per_target_disable = {entry[0]: entry[8] for entry in run_targets}
    per_target_is_mobilint = {entry[0]: entry[9] for entry in run_targets}
    assert per_target_disable == {
        "mobilint/Llama-3.1-8B-Instruct-Batch16": False,
        "meta-llama/Llama-3.1-8B-Instruct": True,
    }
    assert per_target_is_mobilint == {
        "mobilint/Llama-3.1-8B-Instruct-Batch16": True,
        "meta-llama/Llama-3.1-8B-Instruct": False,
    }


def test_collect_text_run_targets_original_models_non_mobilint_only_still_drops(monkeypatch, tmp_path) -> None:
    """Verify --original-models with only a non-Mobilint --model keeps the historical parents-only behavior."""
    monkeypatch.setattr(
        text_bench,
        "_resolve_original_model_ids",
        lambda model_ids: ["meta-llama/Llama-3.1-8B-Instruct"],
    )
    _stub_text_target_hub_probes(
        monkeypatch,
        max_batch_size_map={"meta-llama/Llama-3.1-8B-Instruct": 1},
    )

    args = text_bench._build_arg_parser().parse_args(
        [
            "measure",
            "--original-models",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--output-dir",
            str(tmp_path),
        ]
    )

    _, run_targets = text_bench._collect_text_run_targets(args)

    model_ids = [entry[0] for entry in run_targets]
    assert model_ids == ["meta-llama/Llama-3.1-8B-Instruct"]
    assert not any(mid.startswith("mobilint/") for mid in model_ids)
    assert run_targets[0][8] is True
    assert run_targets[0][9] is False


def test_collect_text_run_targets_mxq_dir_still_ignores_original_models(monkeypatch, tmp_path) -> None:
    """Verify --mxq-dir short-circuits --original-models: no Hub resolve, no merge, just local MXQs."""
    mxq_dir = tmp_path / "mxqs"
    mxq_dir.mkdir()
    (mxq_dir / "mobilint__local-model-W8.mxq").write_bytes(b"")

    resolve_calls: list[list[str]] = []

    def _fail_resolve(model_ids):
        resolve_calls.append(list(model_ids))
        raise AssertionError("_resolve_original_model_ids must not run under --mxq-dir")

    monkeypatch.setattr(text_bench, "_resolve_original_model_ids", _fail_resolve)
    _stub_text_target_hub_probes(monkeypatch, max_batch_size_map={"mobilint/local-model": 1})
    monkeypatch.setattr(
        text_bench,
        "list_default_model_ids",
        lambda task, *, include_private=False: ["mobilint/local-model"],
    )
    monkeypatch.setattr(
        text_bench,
        "_iter_targets_from_mxq_dir",
        lambda *, mxq_dir, available_model_ids: [
            ("mobilint/local-model", ["W8"], "mobilint/local-model", "mobilint_local-model", str(mxq_dir / "m.mxq")),
        ],
    )

    args = text_bench._build_arg_parser().parse_args(
        [
            "measure",
            "--original-models",
            "--mxq-dir",
            str(mxq_dir),
            "--model",
            "mobilint/local-model",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    _, run_targets = text_bench._collect_text_run_targets(args)

    assert resolve_calls == []
    assert [entry[0] for entry in run_targets] == ["mobilint/local-model"]
    assert run_targets[0][8] is False
    assert run_targets[0][9] is True


def test_build_pipeline_forwards_dev_no_only_for_mobilint(monkeypatch) -> None:
    """Verify --dev-no is injected on Mobilint targets and dropped for original ones."""
    captured: dict[str, Any] = {}

    def _fake_pipeline(**kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(text_bench, "hf_pipeline", _fake_pipeline)

    text_bench._build_pipeline(
        "mobilint/mock-batch",
        device="cpu",
        core_mode="single",
        default_single_target_cores=None,
        dev_no=[0, 1],
        max_batch_size=32,
    )
    mobilint_model_kwargs = captured["kwargs"].get("model_kwargs", {})
    assert mobilint_model_kwargs.get("dev_no") == [0, 1]
    assert mobilint_model_kwargs.get("max_batch_size") == 32

    captured.clear()
    text_bench._build_pipeline(
        "upstream/original",
        device="cuda:0",
        core_mode=None,
        dev_no=[0, 1],
        max_batch_size=32,
    )
    original_model_kwargs = captured["kwargs"].get("model_kwargs", {})
    assert "dev_no" not in original_model_kwargs
    assert "max_batch_size" not in original_model_kwargs


def test_text_measure_continues_on_cuda_oom(monkeypatch, tmp_path) -> None:
    """Verify a CUDA OOM at pipeline construction is logged as a skipped row and the loop continues."""
    args = text_bench._build_arg_parser().parse_args(
        [
            "measure",
            "--batch",
            "--original-models",
            "--batch-size",
            "64",
            "--output-dir",
            str(tmp_path),
        ]
    )

    monkeypatch.setattr(
        text_bench,
        "_collect_text_run_targets",
        lambda args: (
            str(tmp_path),
            [
                (
                    "upstream/model-a",
                    [None],
                    "upstream/model-a",
                    "upstream_model-a",
                    None,
                    None,
                    64,
                    "batch",
                    True,
                    False,
                ),
                (
                    "upstream/model-b",
                    [None],
                    "upstream/model-b",
                    "upstream_model-b",
                    None,
                    None,
                    64,
                    "batch",
                    True,
                    False,
                ),
            ],
        ),
    )
    monkeypatch.setattr(text_bench, "_collect_host_pc_info", lambda results_dir: None)
    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_should_precheck_cuda", lambda args: False)

    class _FakeMeasurer:
        def __init__(self, pipeline) -> None:
            pass

        def measure(self, **kwargs):
            return text_bench.BenchmarkResult(
                prefill_sweep=text_bench.SweepData(x_values=[128], tps_values=[10.0], time_values=[0.1]),
                decode_sweep=text_bench.SweepData(x_values=[32], tps_values=[20.0], time_values=[0.2]),
            )

    calls: list[str] = []

    def _fake_build_pipeline(model_id, **kwargs):
        calls.append(model_id)
        if model_id == "upstream/model-a":
            raise RuntimeError("CUDA out of memory. Tried to allocate 10 GiB.")
        return object()

    monkeypatch.setattr(text_bench, "_build_pipeline", _fake_build_pipeline)

    class _FakeTPSMeasurer:
        def __init__(self, pipeline) -> None:
            pass

        def measure(self, **kwargs):
            class _Row:
                prefill_latency = 0.1
                decode_duration = 0.1
                total_time = 0.2
                prefill_tps = 10.0
                decode_tps = 20.0
                prefill_npu_latency_pct = None
                decode_npu_latency_pct = None
                ttft_ms = None

            return _Row()

    class _RowDict(dict):
        pass

    def _fake_asdict(run):
        return {
            "prefill_latency": 0.1,
            "decode_duration": 0.1,
            "total_time": 0.2,
            "prefill_tps": 10.0,
            "decode_tps": 20.0,
            "prefill_npu_latency_pct": None,
            "decode_npu_latency_pct": None,
        }

    monkeypatch.setattr(text_bench, "TPSMeasurer", _FakeTPSMeasurer)
    monkeypatch.setattr(text_bench, "asdict", _fake_asdict)
    monkeypatch.setattr(text_bench, "_build_phase_trackers", lambda args, pipeline: (None, None))
    monkeypatch.setattr(text_bench, "start_qbruntime_trace", lambda path: None)
    monkeypatch.setattr(text_bench, "stop_qbruntime_trace", lambda handle: None)
    monkeypatch.setattr(text_bench, "_release_pipeline", lambda pipeline, device: None)
    monkeypatch.setattr(text_bench, "_clear_cuda_memory", lambda device: None)
    monkeypatch.setattr(text_bench, "_is_cuda_device", lambda device: False)

    assert text_bench._run_measure(args) == 0
    assert calls == ["upstream/model-a", "upstream/model-b"]

    csv_rows = list(csv.DictReader((tmp_path / "combined_measure.csv").open(encoding="utf-8")))
    skipped_rows = [row for row in csv_rows if row.get("skipped_reason") == "cuda_oom"]
    assert len(skipped_rows) == 1
    assert skipped_rows[0]["model"] == "upstream/model-a"


def test_text_measure_continues_on_npu_alloc_error(monkeypatch, tmp_path) -> None:
    """Verify Mobilint NPU allocation failures are logged as skipped rows and the loop continues."""
    from mblt_model_zoo.utils.npu_backend import MobilintBackendAllocError

    args = text_bench._build_arg_parser().parse_args(
        [
            "measure",
            "--batch",
            "--batch-size",
            "64",
            "--dev-no",
            "0,1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    monkeypatch.setattr(
        text_bench,
        "_collect_text_run_targets",
        lambda args: (
            str(tmp_path),
            [
                (
                    "mobilint/model-a",
                    [None],
                    "mobilint/model-a",
                    "mobilint_model-a",
                    None,
                    "single",
                    64,
                    "batch",
                    False,
                    True,
                ),
                (
                    "mobilint/model-b",
                    [None],
                    "mobilint/model-b",
                    "mobilint_model-b",
                    None,
                    "single",
                    64,
                    "batch",
                    False,
                    True,
                ),
            ],
        ),
    )
    monkeypatch.setattr(text_bench, "_collect_host_pc_info", lambda results_dir: None)
    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_should_precheck_cuda", lambda args: False)
    monkeypatch.setattr(text_bench, "_release_pipeline", lambda pipeline, device: None)
    monkeypatch.setattr(text_bench, "_clear_cuda_memory", lambda device: None)
    monkeypatch.setattr(text_bench, "_is_cuda_device", lambda device: False)

    calls: list[str] = []

    def _fake_build_pipeline(model_id, **kwargs):
        calls.append(model_id)
        if model_id == "mobilint/model-a":
            raise MobilintBackendAllocError(
                phase="create",
                slot=3,
                dev=1,
                succeeded_so_far=3,
                n_total=4,
                max_batch_size=64,
                k_per_model=16,
                original=RuntimeError("BadAlloc"),
            )
        return object()

    monkeypatch.setattr(text_bench, "_build_pipeline", _fake_build_pipeline)

    class _FakeTPSMeasurer:
        def __init__(self, pipeline) -> None:
            pass

        def measure(self, **kwargs):
            class _Row:
                prefill_latency = 0.1
                decode_duration = 0.1
                total_time = 0.2
                prefill_tps = 10.0
                decode_tps = 20.0
                prefill_npu_latency_pct = None
                decode_npu_latency_pct = None

            return _Row()

    def _fake_asdict(run):
        return {
            "prefill_latency": 0.1,
            "decode_duration": 0.1,
            "total_time": 0.2,
            "prefill_tps": 10.0,
            "decode_tps": 20.0,
            "prefill_npu_latency_pct": None,
            "decode_npu_latency_pct": None,
        }

    monkeypatch.setattr(text_bench, "TPSMeasurer", _FakeTPSMeasurer)
    monkeypatch.setattr(text_bench, "asdict", _fake_asdict)
    monkeypatch.setattr(text_bench, "_build_phase_trackers", lambda args, pipeline: (None, None))
    monkeypatch.setattr(text_bench, "start_qbruntime_trace", lambda path: None)
    monkeypatch.setattr(text_bench, "stop_qbruntime_trace", lambda handle: None)

    assert text_bench._run_measure(args) == 0
    assert calls == ["mobilint/model-a", "mobilint/model-b"]

    csv_rows = list(csv.DictReader((tmp_path / "combined_measure.csv").open(encoding="utf-8")))
    skipped_rows = [row for row in csv_rows if row.get("skipped_reason") == "npu_alloc"]
    assert len(skipped_rows) == 1
    assert skipped_rows[0]["model"] == "mobilint/model-a"


def test_text_handle_cuda_precheck_skip_writes_structured_record(tmp_path) -> None:
    """Verify the precheck handler records a structured cuda_precheck row and sidecar entry.

    Fresh-failure precedence: the handler stamps ``recorded_at`` on the record
    so a stale on-disk success payload for the same label is masked by rebuild
    reconciliation via timestamp comparison.
    """
    skipped_records: list[dict[str, Any]] = []
    before = time.time()

    text_bench._handle_cuda_precheck_skip(
        label="upstream/model-a",
        device="cuda:0",
        batch_size=32,
        free_bytes=1_000_000,
        required_bytes=8_000_000,
        estimated_bytes=7_000_000,
        skipped_records=skipped_records,
        phase="load",
        output_dir=tmp_path,
        benchmark_type="measure",
    )
    after = time.time()

    assert len(skipped_records) == 1
    record = skipped_records[0]
    assert record["model"] == "upstream/model-a"
    assert record["skipped_reason"] == "cuda_precheck"
    assert record["phase"] == "load"
    assert record["device"] == "cuda:0"
    assert record["batch_size"] == 32
    assert record["free_bytes"] == 1_000_000
    assert record["required_bytes"] == 8_000_000
    assert record["estimated_weights_bytes"] == 7_000_000
    assert "CUDA pre-check VRAM insufficient" in record["detail"]
    assert before <= float(record["recorded_at"]) <= after

    persisted = text_bench._read_skipped_sidecar(tmp_path, "measure")
    assert persisted == skipped_records


def test_text_measure_continues_on_cuda_precheck(monkeypatch, tmp_path) -> None:
    """Verify a CUDA pre-check failure is logged as a skipped row and the loop continues.

    Regression guard for the PR#109 review: a target that fails the pre-check
    used to print and ``continue`` silently, disappearing from combined output
    while the runtime-OOM path recorded a sidecar row. Both should now surface
    a structured skip row.
    """
    args = text_bench._build_arg_parser().parse_args(
        [
            "measure",
            "--batch",
            "--original-models",
            "--batch-size",
            "64",
            "--output-dir",
            str(tmp_path),
        ]
    )

    monkeypatch.setattr(
        text_bench,
        "_collect_text_run_targets",
        lambda args: (
            str(tmp_path),
            [
                (
                    "upstream/model-a",
                    [None],
                    "upstream/model-a",
                    "upstream_model-a",
                    None,
                    None,
                    64,
                    "batch",
                    True,
                    False,
                ),
                (
                    "upstream/model-b",
                    [None],
                    "upstream/model-b",
                    "upstream_model-b",
                    None,
                    None,
                    64,
                    "batch",
                    True,
                    False,
                ),
            ],
        ),
    )
    monkeypatch.setattr(text_bench, "_collect_host_pc_info", lambda results_dir: None)
    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_should_precheck_cuda", lambda args: True)
    # Only model-a exceeds available VRAM; model-b's estimate is unknown so pre-check bails cleanly.
    monkeypatch.setattr(
        text_bench,
        "_estimate_model_weight_bytes",
        lambda model_id, revision: 10 * 1024**3 if model_id == "upstream/model-a" else None,
    )
    monkeypatch.setattr(text_bench, "_cuda_memory_info", lambda device: (1 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(text_bench, "_clear_cuda_memory", lambda device: None)
    monkeypatch.setattr(text_bench, "_is_cuda_device", lambda device: False)

    calls: list[str] = []

    def _fake_build_pipeline(model_id, **kwargs):
        calls.append(model_id)
        return object()

    monkeypatch.setattr(text_bench, "_build_pipeline", _fake_build_pipeline)

    class _FakeTPSMeasurer:
        def __init__(self, pipeline) -> None:
            pass

        def measure(self, **kwargs):
            class _Row:
                prefill_latency = 0.1
                decode_duration = 0.1
                total_time = 0.2
                prefill_tps = 10.0
                decode_tps = 20.0
                prefill_npu_latency_pct = None
                decode_npu_latency_pct = None
                ttft_ms = None

            return _Row()

    def _fake_asdict(run):
        return {
            "prefill_latency": 0.1,
            "decode_duration": 0.1,
            "total_time": 0.2,
            "prefill_tps": 10.0,
            "decode_tps": 20.0,
            "prefill_npu_latency_pct": None,
            "decode_npu_latency_pct": None,
        }

    monkeypatch.setattr(text_bench, "TPSMeasurer", _FakeTPSMeasurer)
    monkeypatch.setattr(text_bench, "asdict", _fake_asdict)
    monkeypatch.setattr(text_bench, "_build_phase_trackers", lambda args, pipeline: (None, None))
    monkeypatch.setattr(text_bench, "start_qbruntime_trace", lambda path: None)
    monkeypatch.setattr(text_bench, "stop_qbruntime_trace", lambda handle: None)
    monkeypatch.setattr(text_bench, "_release_pipeline", lambda pipeline, device: None)

    assert text_bench._run_measure(args) == 0
    # Pre-check must short-circuit model-a before it reaches _build_pipeline.
    assert calls == ["upstream/model-b"]

    csv_rows = list(csv.DictReader((tmp_path / "combined_measure.csv").open(encoding="utf-8")))
    skipped_rows = [row for row in csv_rows if row.get("skipped_reason") == "cuda_precheck"]
    assert len(skipped_rows) == 1
    assert skipped_rows[0]["model"] == "upstream/model-a"

    persisted = text_bench._read_skipped_sidecar(tmp_path, "measure")
    assert [record["model"] for record in persisted] == ["upstream/model-a"]
    assert persisted[0]["skipped_reason"] == "cuda_precheck"
    assert persisted[0]["free_bytes"] == 1 * 1024**3
    assert persisted[0]["estimated_weights_bytes"] == 10 * 1024**3


def test_text_rebuild_measure_charts_preserves_cuda_precheck_skip(tmp_path) -> None:
    """``--rebuild-charts`` must preserve a pre-check skip row across standalone rebuilds.

    Regression guard: task ``679ef`` preserved fresh failures against on-disk
    stale success payloads. Extend the same guarantee to ``cuda_precheck`` so a
    subsequent standalone ``--rebuild-charts`` keeps the pre-check row intact.
    """
    text_bench._write_skipped_sidecar(
        tmp_path,
        [
            {
                "model": "upstream/model-a",
                "device": "cuda:0",
                "batch_size": 64,
                "phase": "load",
                "skipped_reason": "cuda_precheck",
                "free_bytes": 1_073_741_824,
                "required_bytes": 8_589_934_592,
                "estimated_weights_bytes": 7_465_178_776,
                "detail": "CUDA pre-check VRAM insufficient: free=... required=... estimated_weights=...",
            }
        ],
        "measure",
    )

    text_bench._rebuild_measure_outputs(tmp_path)

    persisted = text_bench._read_skipped_sidecar(tmp_path, "measure")
    assert [record["model"] for record in persisted] == ["upstream/model-a"]
    assert persisted[0]["skipped_reason"] == "cuda_precheck"
    rows = list(csv.DictReader((tmp_path / "combined_measure.csv").open("r", encoding="utf-8")))
    skipped_rows = [
        row for row in rows if row.get("model") == "upstream/model-a" and row.get("skipped_reason") == "cuda_precheck"
    ]
    assert len(skipped_rows) == 1


def test_text_rebuild_sweep_charts_preserves_cuda_precheck_skip(tmp_path) -> None:
    """Sweep sibling: ``--rebuild-charts`` must preserve a pre-check skip row for sweep."""
    text_bench._write_skipped_sidecar(
        tmp_path,
        [
            {
                "model": "upstream/sweep-a",
                "device": "cuda:0",
                "batch_size": 16,
                "phase": "load",
                "skipped_reason": "cuda_precheck",
                "free_bytes": 1_073_741_824,
                "required_bytes": 8_589_934_592,
                "estimated_weights_bytes": 7_465_178_776,
            }
        ],
        "sweep",
    )

    text_bench._rebuild_combined_outputs(tmp_path)

    persisted = text_bench._read_skipped_sidecar(tmp_path, "sweep")
    assert [record["model"] for record in persisted] == ["upstream/sweep-a"]
    assert persisted[0]["skipped_reason"] == "cuda_precheck"
    rows = list(csv.DictReader((tmp_path / "combined.csv").open("r", encoding="utf-8")))
    skipped_rows = [
        row for row in rows if row.get("model") == "upstream/sweep-a" and row.get("skipped_reason") == "cuda_precheck"
    ]
    assert len(skipped_rows) == 1


def test_text_target_filter_classifies_caller_mobilint_retained_role(monkeypatch) -> None:
    """Verify --original-models mixed run classifies retained Mobilint sibling role and preserves NPU args."""
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = [
        ("mobilint/Model-A", [None], "mobilint/Model-A", "mobilint_Model-A", None),
        ("meta-llama/Model-A", [None], "meta-llama/Model-A", "meta-llama_Model-A", None),
    ]
    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: False)
    monkeypatch.setattr(
        text_bench,
        "_resolve_config_max_batch_size",
        lambda model_id, revision, *, task: 16 if model_id.startswith("mobilint/") else 1,
    )

    admitted = text_bench._filter_text_targets_by_batch_mode(
        raw_targets,
        batch_mode="batch",
        override_batch_size=64,
        original_models=True,
        caller_model_ids=["mobilint/Model-A"],
        caller_mobilint_ids=["mobilint/Model-A"],
        mxq_dir=None,
    )

    by_id = {target.model_id: target for target in admitted}
    assert by_id["mobilint/Model-A"].is_mobilint is True
    assert by_id["mobilint/Model-A"].role == "caller_mobilint_retained"
    assert by_id["mobilint/Model-A"].disable_npu_specific_args is False
    assert by_id["meta-llama/Model-A"].is_mobilint is False
    assert by_id["meta-llama/Model-A"].role == "resolved_upstream"
    assert by_id["meta-llama/Model-A"].disable_npu_specific_args is True


def test_text_target_filter_classifies_plain_mobilint_role(monkeypatch) -> None:
    """Verify a Mobilint target outside --original-models keeps role=mobilint and disable=False."""
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = [
        ("mobilint/Model-B", [None], "mobilint/Model-B", "mobilint_Model-B", None),
    ]
    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: False)
    monkeypatch.setattr(text_bench, "_resolve_config_max_batch_size", lambda model_id, revision, *, task: 8)

    admitted = text_bench._filter_text_targets_by_batch_mode(
        raw_targets,
        batch_mode="batch",
        override_batch_size=None,
        original_models=False,
        caller_model_ids=["mobilint/Model-B"],
        caller_mobilint_ids=[],
        mxq_dir=None,
    )

    assert len(admitted) == 1
    assert admitted[0].is_mobilint is True
    assert admitted[0].role == "mobilint"
    assert admitted[0].disable_npu_specific_args is False


def test_text_target_filter_classifies_caller_upstream_role(monkeypatch) -> None:
    """Verify a caller-listed non-Mobilint target is caller_upstream, and disable follows --original-models."""
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = [
        ("meta-llama/Solo", [None], "meta-llama/Solo", "meta-llama_Solo", None),
    ]
    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: False)
    monkeypatch.setattr(text_bench, "_resolve_config_max_batch_size", lambda model_id, revision, *, task: 1)

    without_original = text_bench._filter_text_targets_by_batch_mode(
        raw_targets,
        batch_mode="non_batch",
        original_models=False,
        caller_model_ids=["meta-llama/Solo"],
        caller_mobilint_ids=[],
        mxq_dir=None,
    )
    with_original = text_bench._filter_text_targets_by_batch_mode(
        raw_targets,
        batch_mode="non_batch",
        original_models=True,
        caller_model_ids=["meta-llama/Solo"],
        caller_mobilint_ids=[],
        mxq_dir=None,
    )

    assert without_original[0].role == "caller_upstream"
    assert without_original[0].disable_npu_specific_args is False
    assert with_original[0].role == "caller_upstream"
    assert with_original[0].disable_npu_specific_args is True


def test_text_target_filter_mxq_dir_keeps_disable_false(monkeypatch) -> None:
    """Verify --mxq-dir short-circuits disable_npu_specific_args regardless of --original-models."""
    raw_targets: list[tuple[str, list[str | None], str, str, str | None]] = [
        ("mobilint/local", [None], "mobilint/local", "mobilint_local", "/tmp/mxq/mobilint__local-W8.mxq"),
    ]
    monkeypatch.setattr(text_bench, "_select_revision", lambda model_id, candidates: candidates[0])
    monkeypatch.setattr(text_bench, "_has_gguf_artifact", lambda model_id, revision: False)
    monkeypatch.setattr(text_bench, "_resolve_config_max_batch_size", lambda model_id, revision, *, task: 4)

    admitted = text_bench._filter_text_targets_by_batch_mode(
        raw_targets,
        batch_mode="batch",
        original_models=True,
        caller_model_ids=[],
        caller_mobilint_ids=[],
        mxq_dir="/tmp/mxq",
    )

    assert admitted[0].is_mobilint is True
    assert admitted[0].disable_npu_specific_args is False


def test_text_iter_core_modes_for_target_reads_target_disable(monkeypatch) -> None:
    """Verify _iter_core_modes_for_target honors the per-target disable flag, not args.original_models."""
    args = text_bench._build_arg_parser().parse_args(["measure", "--core-mode", "all"])

    disabled = text_bench._iter_core_modes_for_target(args, "non_batch", disable_npu_specific_args=True)
    enabled = text_bench._iter_core_modes_for_target(args, "non_batch", disable_npu_specific_args=False)

    assert disabled == [None]
    assert enabled and None not in enabled


def test_text_args_for_target_device_backend_dev_no_retained_for_mobilint(monkeypatch) -> None:
    """Verify --dev-no is retained on retained Mobilint targets under --original-models mixed runs.

    Regression for the PR-3796045969 review discussion: prior to this refactor, the retained
    Mobilint sibling would still see NPU-specific args disabled at the run-wide level even after
    task 16e4f wired dev_no through per-target. The is_mobilint field now carries the provenance
    without a re-read of ``args.original_models``.
    """
    argv = ["measure", "--original-models", "--model", "mobilint/Model-Z", "--dev-no", "0,1,2,3"]
    args = text_bench._build_arg_parser().parse_args(argv)
    text_bench._resolve_runtime_defaults(args, argv)

    mobilint_args = text_bench._args_for_target_device_backend(
        args,
        model_id="mobilint/Model-Z",
        mxq_path=None,
        is_mobilint=True,
    )
    parent_args = text_bench._args_for_target_device_backend(
        args,
        model_id="meta-llama/Model-Z",
        mxq_path=None,
        is_mobilint=False,
    )

    assert mobilint_args.device == "cpu"
    assert mobilint_args.device_backend == "npu"
    assert mobilint_args.dev_no == [0, 1, 2, 3]
    assert parent_args.device == "cuda"
    assert parent_args.device_backend == "gpu"


def test_benchmark_text_generation_forbids_downstream_original_models_reads() -> None:
    """Static guard: no benchmark_text_generation_models.py code outside the collection/parse
    sites reads ``args.original_models``.

    This is the anti-regression contract for the refactor: every downstream helper (measurement
    loop, sweep loop, device-backend resolver, chunk-size resolver, core-mode iterator) reads
    per-target ``TextBenchmarkTarget`` metadata via the run tuple instead of re-checking the
    global ``--original-models`` flag. If a future review adds another consumer, this guard
    catches it before it lands.
    """
    src = Path(text_bench.__file__).read_text(encoding="utf-8")
    # Whitelist: legitimate reads at the arg-parse/collection layer where the flag is a global
    # run-mode selector, not a per-target policy. Downstream helpers must read TextBenchmarkTarget
    # fields instead.
    allowed_line_prefixes = (
        "        original_models=args.original_models,",  # _resolve_default_device/_backend
        "        if args.models or args.original_models or args.all",  # advisory print
        "        if args.original_models:",  # top-level collection branch
    )
    hits: list[tuple[int, str]] = []
    for lineno, raw in enumerate(src.splitlines(), start=1):
        if "args.original_models" not in raw:
            continue
        stripped = raw.rstrip()
        # Docstring/comment references are safe.
        if stripped.lstrip().startswith("#") or "``args.original_models``" in stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in allowed_line_prefixes):
            continue
        hits.append((lineno, stripped))
    assert not hits, (
        "Downstream args.original_models reads leaked outside the collection/parse sites: "
        f"{hits}. Read TextBenchmarkTarget.is_mobilint / .disable_npu_specific_args instead."
    )
