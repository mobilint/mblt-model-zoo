"""ASR benchmark metric helpers.

This module keeps transcript normalization and aggregate metric math isolated from
the benchmark entry script so they can be unit-tested without requiring model
loading.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from jiwer import cer, wer

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


@dataclass
class SampleTiming:
    """One ASR sample's measured timing and decoding stats."""

    sample_id: str
    audio_duration_s: float
    generate_time_s: float
    num_generated_tokens: int
    num_beams: int | None
    reference: str
    hypothesis: str


@dataclass
class ASRMetricSummary:
    """Aggregated WER/CER and ASR speed metrics for one run."""

    num_samples: int
    total_audio_s: float
    total_generate_s: float
    wer: float
    cer: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    throughput_samples_per_s: float
    rtf: float
    inverse_rtf: float
    decode_tokens_per_s: float
    avg_tokens_per_sample: float


def _safe_div(numerator: float, denominator: float) -> float:
    """Return a safe division result, falling back to zero for empty denominators."""

    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Return a linear-interpolated percentile from pre-sorted values."""

    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = float(sorted_values[lower])
    upper_value = float(sorted_values[upper])
    return lower_value + (upper_value - lower_value) * (rank - lower)


def normalize_transcript(text: str, *, language: str = "en") -> str:
    """Normalize a transcript for WER/CER comparison.

    Args:
        text: Raw transcript text.
        language: Language tag used to select normalization behavior.

    Returns:
        A normalized transcript string.
    """

    normalized = text.casefold().strip()
    if language == "en":
        normalized = normalized.translate(_PUNCT_TRANSLATION)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def compute_wer_cer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    language: str = "en",
) -> tuple[float, float]:
    """Compute normalized WER and CER.

    Args:
        references: Reference transcripts.
        hypotheses: Predicted transcripts.
        language: Language tag used to select normalization behavior.

    Returns:
        Tuple of `(wer, cer)` in 0..1 range.
    """

    normalized_references = [normalize_transcript(text, language=language) for text in references]
    normalized_hypotheses = [normalize_transcript(text, language=language) for text in hypotheses]
    word_error_rate = float(wer(normalized_references, normalized_hypotheses))
    char_error_rate = float(cer(normalized_references, normalized_hypotheses))
    return word_error_rate, char_error_rate


def summarize_timings(timings: Sequence[SampleTiming], *, language: str = "en") -> ASRMetricSummary:
    """Aggregate per-sample timing records into ASR metrics.

    Args:
        timings: Measured per-sample timing records.
        language: Language tag used to select normalization behavior.

    Returns:
        Aggregated benchmark summary.
    """

    if not timings:
        return ASRMetricSummary(
            num_samples=0,
            total_audio_s=0.0,
            total_generate_s=0.0,
            wer=0.0,
            cer=0.0,
            mean_latency_s=0.0,
            p50_latency_s=0.0,
            p95_latency_s=0.0,
            throughput_samples_per_s=0.0,
            rtf=0.0,
            inverse_rtf=0.0,
            decode_tokens_per_s=0.0,
            avg_tokens_per_sample=0.0,
        )

    latencies = sorted(float(item.generate_time_s) for item in timings)
    total_audio_s = sum(float(item.audio_duration_s) for item in timings)
    total_generate_s = sum(float(item.generate_time_s) for item in timings)
    total_tokens = sum(int(item.num_generated_tokens) for item in timings)
    references = [item.reference for item in timings]
    hypotheses = [item.hypothesis for item in timings]
    word_error_rate, char_error_rate = compute_wer_cer(references, hypotheses, language=language)
    rtf = _safe_div(total_generate_s, total_audio_s)

    return ASRMetricSummary(
        num_samples=len(timings),
        total_audio_s=total_audio_s,
        total_generate_s=total_generate_s,
        wer=word_error_rate,
        cer=char_error_rate,
        mean_latency_s=_safe_div(total_generate_s, float(len(timings))),
        p50_latency_s=_percentile(latencies, 0.50),
        p95_latency_s=_percentile(latencies, 0.95),
        throughput_samples_per_s=_safe_div(float(len(timings)), total_generate_s),
        rtf=rtf,
        inverse_rtf=_safe_div(total_audio_s, total_generate_s),
        decode_tokens_per_s=_safe_div(float(total_tokens), total_generate_s),
        avg_tokens_per_sample=_safe_div(float(total_tokens), float(len(timings))),
    )


def format_metrics_row(
    label: str,
    num_beams: int | None,
    summary: ASRMetricSummary,
    device_metric: Mapping[str, float | None],
) -> dict[str, Any]:
    """Flatten one benchmark summary for CSV/Markdown output."""

    row: dict[str, Any] = {
        "model": label,
        "num_beams": num_beams,
        **asdict(summary),
    }
    for key, value in device_metric.items():
        row[key] = value
    return row


def summary_to_dict(summary: ASRMetricSummary) -> dict[str, Any]:
    """Convert a summary dataclass to a JSON-ready dict."""

    return asdict(summary)