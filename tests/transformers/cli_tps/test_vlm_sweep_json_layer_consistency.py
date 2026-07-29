"""Regression tests for VLM tps CLI JSON layer consistency (PR #102 R5).

The three layers ``runs[i]`` / ``aggregate`` / ``summary`` must agree on
canonical schema keys.  These tests pin the three plumbing gaps flagged by
Codex Review round 5:

1. ``runs[i].total`` for VLM measure must include ``vision * batch_size``
   (matching ``summary.total``).
2. Vision sweep ``runs[i]`` must expose the canonical ``vision_encode``
   key in milliseconds (not the seconds-valued ``vision_encode_latency``)
   plus per-run device/energy fields.
3. LLM sweep ``aggregate`` must expose device/energy scalars so consumers
   diffing the three layers see the same key set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mblt_model_zoo.cli import tps as tps_cli
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
    BenchmarkResult,
    SweepData,
)


def _measure_args(*, batch_size: int, json_path: Path) -> argparse.Namespace:
    """Return an argparse.Namespace matching ``_run_vlm_measure``'s expectations."""
    return argparse.Namespace(
        task="image-text-to-text",
        model="dummy",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
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
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        batch_size=batch_size,
        warmup=0,
        repeat=1,
        image_resolution=224,
        prefill=8,
        decode=2,
        prompt="Describe.",
        npu_prefill_chunk_size=None,
        device_metrics=False,
        json=str(json_path),
        device_backend="none",
    )


def _sweep_args(*, batch_size: int, json_path: Path) -> argparse.Namespace:
    """Return an argparse.Namespace matching ``_run_vlm_sweep``'s expectations."""
    return argparse.Namespace(
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
        batch_size=batch_size,
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
        json=str(json_path),
        csv=None,
        plot=None,
    )


def test_vlm_measure_runs_total_matches_summary_at_batch_gt_one(monkeypatch, tmp_path):
    """runs[0].total must equal summary.total.mean at batch_size > 1.

    Without ``obj.batch_size`` on the VLMSingleMeasurement, the schema
    extractor for ``runs[i].total`` falls back to batch_size=1 and diverges
    from ``summary.total`` (which the CLI computes with the real batch size).
    """
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline) -> None:
            del pipeline

        def measure_vision(self, **kwargs):
            del kwargs
            return [(0.1, 10.0)]  # vision_encode_latency=0.1s

        def measure_llm_full(self, **kwargs):
            del kwargs
            result = BenchmarkResult()
            result.prefill_sweep.x_values.append(8)
            result.prefill_sweep.tps_values.append(80.0)
            result.prefill_sweep.time_values.append(0.2)  # prefill_latency=0.2s
            result.prefill_sweep.avg_total_token_latency_values.append(0.025)
            result.prefill_sweep.avg_npu_token_latency_values.append(None)
            result.decode_sweep.x_values.append(8)
            result.decode_sweep.tps_values.append(10.0)
            result.decode_sweep.time_values.append(0.3)  # decode_duration=0.3s
            result.decode_sweep.avg_total_token_latency_values.append(0.15)
            result.decode_sweep.avg_npu_token_latency_values.append(None)
            return result

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: object())
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: None)
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(tps_cli, "_resolve_cli_batch_size", lambda args, pipeline: 4)
    monkeypatch.setattr(benchmark_utils, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)

    json_path = tmp_path / "vlm_measure.json"
    args = _measure_args(batch_size=4, json_path=json_path)

    assert tps_cli._run_vlm_measure(args) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    # vision=0.1s * batch_size=4 + llm.total_time=(0.2+0.3)=0.5s → 0.9s → 900ms
    expected_total_ms = (0.1 * 4 + 0.5) * 1000.0
    assert payload["runs"][0]["total"] == pytest.approx(expected_total_ms)
    assert payload["summary"]["total"]["mean"] == pytest.approx(expected_total_ms)
    assert payload["runs"][0]["total"] == pytest.approx(payload["summary"]["total"]["mean"])


def test_vlm_sweep_vision_runs_use_canonical_keys(monkeypatch, tmp_path):
    """Vision sweep runs must expose ``vision_encode`` in ms (schema key), not raw seconds."""

    class _FakeTracker:
        def start(self) -> None:
            pass

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline) -> None:
            del pipeline

        def measure_vision(self, *args, **kwargs):
            del args, kwargs
            return [(0.1, 10.0)]

        def measure_llm_full(self, *args, **kwargs):
            del args, kwargs
            return BenchmarkResult(
                prefill_sweep=SweepData(
                    x_values=[128, 256], tps_values=[10.0, 20.0], time_values=[0.1, 0.2]
                ),
                decode_sweep=SweepData(
                    x_values=[128, 256, 512],
                    tps_values=[30.0, 40.0, 50.0],
                    time_values=[0.3, 0.4, 0.5],
                ),
            )

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: object())
    monkeypatch.setattr(tps_cli, "_resolve_cli_batch_size", lambda args, pipeline: 2)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: _FakeTracker())
    monkeypatch.setattr(
        tps_cli, "_build_phase_trackers", lambda args, pipeline: (_FakeTracker(), _FakeTracker())
    )
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(tps_cli, "_stop_tracker_safe", lambda tracker: None)
    monkeypatch.setattr(
        tps_cli,
        "_extract_device_metric",
        lambda tracker: {
            "avg_power_w": 10.0,
            "p99_power_w": 12.0,
            "avg_utilization_pct": 80.0,
            "p99_utilization_pct": 95.0,
            "avg_temperature_c": 45.0,
            "p99_temperature_c": 55.0,
            "avg_memory_used_mb": 4096.0,
            "p99_memory_used_mb": 5120.0,
            "total_memory_mb": 16384.0,
            "avg_memory_used_pct": 25.0,
            "p99_memory_used_pct": 31.25,
        },
    )
    monkeypatch.setattr(
        tps_cli,
        "_extract_device_time_series",
        lambda tracker: {
            "power_w": [
                {"timestamp_s": 0.0, "value": 10.0},
                {"timestamp_s": 1.0, "value": 10.0},
            ]
        },
    )
    monkeypatch.setattr(
        "mblt_model_zoo.hf_transformers.utils.benchmark_utils.VLMTPSMeasurer",
        _FakeVLMTPSMeasurer,
    )

    json_path = tmp_path / "vlm_sweep.json"
    args = _sweep_args(batch_size=2, json_path=json_path)

    assert tps_cli._run_vlm_sweep(args) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    vision_res = payload["vision_results"][0]
    assert vision_res["units"]["vision_encode"] == "ms"
    run = vision_res["runs"][0]
    # Canonical key present with ms value ...
    assert "vision_encode" in run
    assert run["vision_encode"] == pytest.approx(100.0)  # 0.1s → 100ms
    # ... and the pre-migration key is gone.
    assert "vision_encode_latency" not in run
    # Per-run energy + device fields must be present now that runs go through
    # the schema.
    assert run["vision_energy"] == pytest.approx(10.0)  # ∫10W dt from 0→1s
    assert run["vision_avg_power"] == pytest.approx(10.0)
    assert run["vision_p99_power"] == pytest.approx(12.0)
    assert run["vision_avg_util"] == pytest.approx(80.0)
    assert run["vision_avg_temp"] == pytest.approx(45.0)
    assert run["vision_avg_mem_used"] == pytest.approx(4096.0)


def test_vlm_sweep_llm_aggregate_has_device_metrics(monkeypatch, tmp_path):
    """LLM sweep aggregate must expose device/energy scalars (not just sweep curves)."""

    class _FakeTracker:
        def start(self) -> None:
            pass

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline) -> None:
            del pipeline

        def measure_vision(self, *args, **kwargs):
            del args, kwargs
            return [(0.1, 10.0)]

        def measure_llm_full(self, *args, **kwargs):
            del args, kwargs
            return BenchmarkResult(
                prefill_sweep=SweepData(
                    x_values=[128, 256], tps_values=[10.0, 20.0], time_values=[0.1, 0.2]
                ),
                decode_sweep=SweepData(
                    x_values=[128, 256, 512],
                    tps_values=[30.0, 40.0, 50.0],
                    time_values=[0.3, 0.4, 0.5],
                ),
            )

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: object())
    monkeypatch.setattr(tps_cli, "_resolve_cli_batch_size", lambda args, pipeline: 2)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: _FakeTracker())
    monkeypatch.setattr(
        tps_cli, "_build_phase_trackers", lambda args, pipeline: (_FakeTracker(), _FakeTracker())
    )
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(tps_cli, "_stop_tracker_safe", lambda tracker: None)
    monkeypatch.setattr(
        tps_cli,
        "_extract_device_metric",
        lambda tracker: {
            "avg_power_w": 10.0,
            "p99_power_w": 12.0,
            "avg_utilization_pct": 80.0,
            "p99_utilization_pct": 95.0,
            "avg_temperature_c": 45.0,
            "p99_temperature_c": 55.0,
            "avg_memory_used_mb": 4096.0,
            "p99_memory_used_mb": 5120.0,
            "total_memory_mb": 16384.0,
            "avg_memory_used_pct": 25.0,
            "p99_memory_used_pct": 31.25,
        },
    )
    monkeypatch.setattr(
        tps_cli,
        "_extract_device_time_series",
        lambda tracker: {
            "power_w": [
                {"timestamp_s": 0.0, "value": 10.0},
                {"timestamp_s": 1.0, "value": 10.0},
            ]
        },
    )
    monkeypatch.setattr(
        "mblt_model_zoo.hf_transformers.utils.benchmark_utils.VLMTPSMeasurer",
        _FakeVLMTPSMeasurer,
    )

    json_path = tmp_path / "vlm_sweep.json"
    args = _sweep_args(batch_size=2, json_path=json_path)

    assert tps_cli._run_vlm_sweep(args) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    aggregate = payload["llm_results"]["aggregate"]

    # Power / util / temp / mem scalars (mean-across-runs) must land here now.
    assert aggregate["llm_avg_power"] == pytest.approx(10.0)
    assert aggregate["llm_p99_power"] == pytest.approx(12.0)
    assert aggregate["llm_avg_util"] == pytest.approx(80.0)
    assert aggregate["llm_avg_temp"] == pytest.approx(45.0)
    assert aggregate["llm_avg_mem_used"] == pytest.approx(4096.0)

    # Energy + efficiency come from the shared text-sweep helper — verify
    # both endpoints are populated on aggregate.
    assert aggregate["llm_prefill_energy"] == pytest.approx(10.0)
    assert aggregate["llm_decode_energy"] == pytest.approx(10.0)
    assert aggregate["llm_total_energy"] == pytest.approx(20.0)
    # (128+256)*2 / 10 J
    assert aggregate["llm_prefill_tps_per_w"] == pytest.approx((128 + 256) * 2 / 10.0)
    # 32*3*2 / 10 J
    assert aggregate["llm_decode_tps_per_w"] == pytest.approx(32 * 3 * 2 / 10.0)
