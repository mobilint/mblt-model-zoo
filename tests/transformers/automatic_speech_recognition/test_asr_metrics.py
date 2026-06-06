from benchmark.transformers.asr_metrics import SampleTiming, compute_wer_cer, normalize_transcript, summarize_timings


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