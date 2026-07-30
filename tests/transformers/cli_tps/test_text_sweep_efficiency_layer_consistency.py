"""Regression test for text TPS sweep JSON efficiency layer consistency.

PR #102 Codex Review round 6 flagged that ``summary.prefill_tps_per_w`` and
its siblings were computed from last-sweep-point TPS divided by
phase-average power, while ``runs[i].prefill_tps_per_w`` and
``aggregate.prefill_tps_per_w`` were computed from total swept tokens
divided by phase-integrated energy.  When TPS varies across sweep points
(prefill length, cache length), the two definitions diverge silently.

This test pins the invariant that all three layers now share the same
per-run phase-wide definition.
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
    SingleMeasurement,
)


class _DummyConfig:
    def __init__(self, max_batch_size: int) -> None:
        self.max_batch_size = max_batch_size


class _FakePhaseTracker:
    """Emit a phase-average power and a 1-second constant-power trace."""

    def __init__(self, avg_power_w: float, energy_j: float) -> None:
        self._avg_power_w = avg_power_w
        # A 1-second constant trace at ``energy_j`` watts integrates
        # trapezoidally to ``energy_j`` joules.
        self._trace = [(0.0, energy_j), (1.0, energy_j)]

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_metric(self) -> dict[str, float]:
        return {"avg_power_w": self._avg_power_w, "p99_power_w": self._avg_power_w}

    def get_total_power_trace(self) -> list[tuple[float, float]]:
        return list(self._trace)


def _sweep_args(*, json_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        task="text-generation",
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
        batch_size=1,
        warmup=0,
        repeat=1,
        prefill_range=(8, 16, 8),
        cache_lengths=[8, 16],
        decode_window=2,
        npu_prefill_chunk_size=None,
        trace=None,
        device_metrics=True,
        json=str(json_path),
        csv=None,
        plot=None,
        device_backend="npu",
    )


def test_text_sweep_efficiency_summary_matches_runs(monkeypatch, tmp_path):
    """summary.<efficiency> == runs[0].<efficiency> across all four keys.

    Constructs a sweep with two points whose TPS values differ by 5x
    (10 -> 50), so a last-point/phase-average-power calculation would
    diverge from the whole-sweep phase-wide calculation the runs and
    aggregate carry.  Before the fix, summary used the former; this
    test now pins it to the latter, matching runs/aggregate.
    """
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=1)))

    class _FakeTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure(self, **kwargs) -> SingleMeasurement:
            return SingleMeasurement(
                num_prefill=kwargs["num_prefill"],
                num_decode=kwargs["num_decode"],
                prefill_latency=1.0,
                prefill_tps=1.0,
                decode_duration=1.0,
                decode_tps=1.0,
                total_time=2.0,
                avg_total_prefill_token_latency=1.0,
                avg_npu_prefill_token_latency=None,
                avg_total_decode_token_latency=1.0,
                avg_npu_decode_token_latency=None,
            )

        def measure_full(self, **kwargs) -> BenchmarkResult:
            # Fire the phase hooks so the fake trackers get start/stop calls
            # in the same order the real measurer produces.
            if kwargs.get("on_prefill_start") is not None:
                kwargs["on_prefill_start"]()
                kwargs["on_prefill_end"]()
                kwargs["on_decode_start"]()
                kwargs["on_decode_end"]()
            result = BenchmarkResult()
            # Two sweep points with materially different TPS ensure a
            # last-point-based definition would diverge from a whole-sweep
            # one.
            result.prefill_sweep.x_values.extend([8, 16])
            result.prefill_sweep.tps_values.extend([10.0, 50.0])
            result.prefill_sweep.time_values.extend([0.8, 0.32])
            result.prefill_sweep.avg_total_token_latency_values.extend([0.1, 0.02])
            result.prefill_sweep.avg_npu_token_latency_values.extend([None, None])
            result.decode_sweep.x_values.extend([8, 16])
            result.decode_sweep.tps_values.extend([10.0, 50.0])
            result.decode_sweep.time_values.extend([0.2, 0.04])
            result.decode_sweep.avg_total_token_latency_values.extend([0.1, 0.02])
            result.decode_sweep.avg_npu_token_latency_values.extend([None, None])
            return result

        def plot_and_save(self, result, save_path) -> None:
            del result, save_path

    def _fake_build_phase_trackers(args, pipeline):
        del args, pipeline
        # avg_power=4W, energy=4J: OLD prefill_tps_per_w=50/4=12.5,
        # NEW=(8+16)/4=6.0.
        prefill = _FakePhaseTracker(avg_power_w=4.0, energy_j=4.0)
        # avg_power=2W, energy=2J: OLD decode_tps_per_w=50/2=25.0,
        # NEW=(2*2)/2=2.0.
        decode = _FakePhaseTracker(avg_power_w=2.0, energy_j=2.0)
        return prefill, decode

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_phase_trackers", _fake_build_phase_trackers)
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "TPSMeasurer", _FakeTPSMeasurer)

    json_path = tmp_path / "text_sweep.json"
    args = _sweep_args(json_path=json_path)

    assert tps_cli._run_text_sweep(args) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    runs = payload["runs"]
    aggregate = payload["aggregate"]
    summary = payload["summary"]

    # Whole-sweep, phase-integrated definition (batch_size=1, repeat=1).
    expected_prefill_tps_per_w = (8 + 16) / 4.0  # 24 tok / 4 J
    expected_decode_tps_per_w = (2 * 2) / 2.0  # decode_window * points / energy
    expected_prefill_j_per_token = 4.0 / (8 + 16)
    expected_decode_j_per_token = 2.0 / (2 * 2)

    assert runs[0]["prefill_tps_per_w"] == pytest.approx(expected_prefill_tps_per_w)
    assert runs[0]["decode_tps_per_w"] == pytest.approx(expected_decode_tps_per_w)
    assert runs[0]["prefill_j_per_tok"] == pytest.approx(expected_prefill_j_per_token)
    assert runs[0]["decode_j_per_tok"] == pytest.approx(expected_decode_j_per_token)

    # Cross-layer: runs[0] and aggregate already agree today (pin it).
    assert aggregate["prefill_tps_per_w"] == pytest.approx(runs[0]["prefill_tps_per_w"])
    assert aggregate["decode_tps_per_w"] == pytest.approx(runs[0]["decode_tps_per_w"])
    assert aggregate["prefill_j_per_tok"] == pytest.approx(runs[0]["prefill_j_per_tok"])
    assert aggregate["decode_j_per_tok"] == pytest.approx(runs[0]["decode_j_per_tok"])

    # Invariant this fix establishes: summary now matches runs/aggregate.
    assert summary["prefill_tps_per_w"]["mean"] == pytest.approx(runs[0]["prefill_tps_per_w"])
    assert summary["decode_tps_per_w"]["mean"] == pytest.approx(runs[0]["decode_tps_per_w"])
    assert summary["prefill_j_per_tok"]["mean"] == pytest.approx(runs[0]["prefill_j_per_tok"])
    assert summary["decode_j_per_tok"]["mean"] == pytest.approx(runs[0]["decode_j_per_tok"])

    # Sanity: the OLD (buggy) last-point/phase-avg-power formula would have
    # produced materially different numbers, so this test would fail before
    # the fix.
    old_prefill_tps_per_w = 50.0 / 4.0
    old_decode_tps_per_w = 50.0 / 2.0
    assert summary["prefill_tps_per_w"]["mean"] != pytest.approx(old_prefill_tps_per_w)
    assert summary["decode_tps_per_w"]["mean"] != pytest.approx(old_decode_tps_per_w)
