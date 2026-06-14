import builtins

import pytest

from benchmark.transformers.asr_metrics import (
    SampleTiming,
    add_device_efficiency_metrics,
    compute_wer_cer,
    normalize_transcript,
    summarize_timings,
)


def test_normalize_transcript_basic() -> None:
    """Verify transcript normalization removes case, punctuation, and duplicate spaces."""

    assert normalize_transcript(" Hello,   WORLD!! ") == "hello world"


def test_compute_wer_cer_perfect_match() -> None:
    """Verify identical transcripts produce zero error."""

    word_error_rate, char_error_rate = compute_wer_cer(["this is a test"], ["this is a test"])

    assert word_error_rate == 0.0
    assert char_error_rate == 0.0


def test_compute_wer_cer_known_pair() -> None:
    """Verify WER/CER for a known mismatch pair remain stable."""

    word_error_rate, char_error_rate = compute_wer_cer(["this is a test"], ["this was a test"])

    assert word_error_rate == 0.25
    assert char_error_rate == 0.14285714285714285


def test_compute_wer_cer_respects_language_normalization() -> None:
    """Verify punctuation normalization depends on the requested language."""

    assert compute_wer_cer(["hello world"], ["hello, world"], language="en") == (0.0, 0.0)

    word_error_rate, char_error_rate = compute_wer_cer(["hello world"], ["hello, world"], language="ko")

    assert word_error_rate > 0.0
    assert char_error_rate > 0.0


def test_compute_wer_cer_missing_jiwer_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify lazy jiwer import raises an actionable dependency message."""

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "jiwer":
            raise ModuleNotFoundError("No module named 'jiwer'", name="jiwer")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="uv sync --group dev"):
        compute_wer_cer(["hello world"], ["hello world"])


def test_summarize_timings_aggregation() -> None:
    """Verify timing aggregation computes latency and throughput correctly."""

    timings = [
        SampleTiming(
            sample_id="a",
            audio_duration_s=2.0,
            generate_time_s=1.0,
            num_generated_tokens=10,
            num_beams=1,
            reference="hello world",
            hypothesis="hello world",
        ),
        SampleTiming(
            sample_id="b",
            audio_duration_s=3.0,
            generate_time_s=2.0,
            num_generated_tokens=20,
            num_beams=1,
            reference="foo bar",
            hypothesis="foo bar",
        ),
    ]

    summary = summarize_timings(timings)

    assert summary.num_samples == 2
    assert summary.total_audio_s == 5.0
    assert summary.total_generate_s == 3.0
    assert summary.mean_latency_s == 1.5
    assert summary.p50_latency_s == 1.5
    assert summary.throughput_samples_per_s == 2.0 / 3.0
    assert summary.rtf == 3.0 / 5.0
    assert summary.inverse_rtf == 5.0 / 3.0
    assert summary.decode_tokens_per_s == 10.0
    assert summary.avg_tokens_per_sample == 15.0


def test_summarize_timings_respects_language_normalization() -> None:
    """Verify aggregate metrics pass the requested language into transcript normalization."""

    timings = [
        SampleTiming(
            sample_id="a",
            audio_duration_s=1.0,
            generate_time_s=0.5,
            num_generated_tokens=2,
            num_beams=None,
            reference="hello world",
            hypothesis="hello, world",
        )
    ]

    english_summary = summarize_timings(timings, language="en")
    korean_summary = summarize_timings(timings, language="ko")

    assert english_summary.wer == 0.0
    assert english_summary.cer == 0.0
    assert korean_summary.wer > 0.0
    assert korean_summary.cer > 0.0


def test_add_device_efficiency_metrics_computes_asr_power_metrics() -> None:
    """Verify ASR energy-efficiency metrics are derived from audio and energy."""

    metrics = add_device_efficiency_metrics(
        {"total_audio_s": 12.0, "rtf": 0.5},
        {"total_energy_j": 3.0, "avg_power_w": 10.0},
    )

    assert metrics["sec_per_j"] == 4.0
    assert metrics["j_per_sec"] == 0.25


def test_add_device_efficiency_metrics_handles_missing_device_values() -> None:
    """Verify ASR energy-efficiency metrics stay empty without usable power or energy inputs."""

    metrics = add_device_efficiency_metrics(
        {"total_audio_s": 12.0, "rtf": 0.5},
        {"total_energy_j": 0.0, "avg_power_w": None},
    )

    assert metrics["sec_per_j"] is None
    assert metrics["j_per_sec"] is None