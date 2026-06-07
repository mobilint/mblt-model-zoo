from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

try:
    from benchmark.transformers.chart_utils import (
        default_charts_dir,
        folder_labels,
        plot_scalar_chart,
        plot_token_chart,
    )
except ModuleNotFoundError:
    from chart_utils import default_charts_dir, folder_labels, plot_scalar_chart, plot_token_chart


@dataclass(frozen=True)
class ScalarChartSpec:
    """One scalar chart specification."""

    filename: str
    title: str
    x_label: str
    attr: str


@dataclass(frozen=True)
class TokenChartSpec:
    """One token-keyed chart specification."""

    filename: str
    title: str
    x_label: str
    attr: str


@dataclass
class BaseCompareMetric(ABC):
    """Common compare metric base shared by benchmark tasks."""

    avg_power_w: float | None = None
    p99_power_w: float | None = None
    total_energy_j: float | None = None
    avg_utilization_pct: float | None = None
    p99_utilization_pct: float | None = None
    avg_temperature_c: float | None = None
    p99_temperature_c: float | None = None
    avg_memory_used_mb: float | None = None
    p99_memory_used_mb: float | None = None
    avg_memory_used_pct: float | None = None
    p99_memory_used_pct: float | None = None

    TASK: ClassVar[str] = ""
    SCALAR_SPECS: ClassVar[Sequence[ScalarChartSpec]] = ()
    TOKEN_SPECS: ClassVar[Sequence[TokenChartSpec]] = ()

    @classmethod
    @abstractmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BaseCompareMetric | None:
        """Build a compare metric from one benchmark JSON payload."""

    @classmethod
    def shared_scalar_specs(cls) -> Sequence[ScalarChartSpec]:
        """Return scalar chart specs shared across tasks."""

        return _SHARED_SCALAR_SPECS


_SHARED_SCALAR_SPECS: tuple[ScalarChartSpec, ...] = (
    ScalarChartSpec("avg_power_w.png", "Power", "Power (Watts)", "avg_power_w"),
    ScalarChartSpec("avg_temperature_c.png", "Temperature", "Temperature (Celsius)", "avg_temperature_c"),
    ScalarChartSpec("avg_utilization_pct.png", "Utilization", "Utilization (Percent)", "avg_utilization_pct"),
    ScalarChartSpec(
        "avg_memory_used_mb.png",
        "Memory Used Megabytes",
        "Memory Used (Megabytes)",
        "avg_memory_used_mb",
    ),
    ScalarChartSpec("total_energy_j.png", "Total Energy", "Energy (Joules)", "total_energy_j"),
)

_BEAM_SUFFIX_RE = re.compile(r"_beams(?:\d+|default)$")


def _as_float(value: Any) -> float | None:
    """Return a float for numeric values."""

    return float(value) if isinstance(value, (int, float)) else None


def _summary_mean(mapping: Mapping[str, Any], key: str) -> float | None:
    """Return ``summary[key].mean`` as a float when present."""

    value = mapping.get(key)
    if not isinstance(value, Mapping):
        return None
    return _as_float(value.get("mean"))


def _strip_group_id(model_id: str) -> str:
    """Compare by model id only, ignoring a leading group id prefix."""

    return model_id.split("__", 1)[1] if "__" in model_id else model_id


def normalize_model_key(path: Path, loaded_model_id: str) -> str:
    """Normalize a model id for cross-folder comparison."""

    stem = path.stem
    if "__" in stem:
        return _BEAM_SUFFIX_RE.sub("", _strip_group_id(stem))
    key = _BEAM_SUFFIX_RE.sub("", _strip_group_id(loaded_model_id))
    if "/" in key:
        key = key.split("/", 1)[1]
    return key


@dataclass
class LLMCompareMetric(BaseCompareMetric):
    """Text-generation compare metric."""

    prefill_tps: dict[int, float] = field(default_factory=dict)
    decode_tps: dict[int, float] = field(default_factory=dict)
    prefill_latency_ms: dict[int, float] = field(default_factory=dict)
    decode_duration_ms: dict[int, float] = field(default_factory=dict)
    prefill_tokens_per_j: float | None = None
    decode_tokens_per_j: float | None = None
    prefill_j_per_token: float | None = None
    decode_j_per_token: float | None = None

    TASK: ClassVar[str] = "text-generation"
    TOKEN_SPECS: ClassVar[Sequence[TokenChartSpec]] = (
        TokenChartSpec("prefill_tps.png", "Prefill Tokens Per Second", "Tokens Per Second", "prefill_tps"),
        TokenChartSpec("decode_tps.png", "Decode Tokens Per Second", "Tokens Per Second", "decode_tps"),
    )
    SCALAR_SPECS: ClassVar[Sequence[ScalarChartSpec]] = (
        ScalarChartSpec(
            "prefill_tokens_per_j.png",
            "Prefill Tokens Per Joule",
            "Tokens Per Joule",
            "prefill_tokens_per_j",
        ),
        ScalarChartSpec(
            "decode_tokens_per_j.png",
            "Decode Tokens Per Joule",
            "Tokens Per Joule",
            "decode_tokens_per_j",
        ),
    )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LLMCompareMetric | None:
        """Parse one text-generation benchmark payload."""

        benchmark = payload.get("benchmark", payload)
        if not isinstance(benchmark, Mapping):
            return None
        prefill = benchmark.get("prefill_sweep", {})
        decode = benchmark.get("decode_sweep", {})
        if not isinstance(prefill, Mapping) or not isinstance(decode, Mapping):
            return None
        device = payload.get("device", {})
        if not isinstance(device, Mapping):
            device = {}

        def _token_map(phase: Mapping[str, Any], value_key: str, *, ms: bool = False) -> dict[int, float]:
            out: dict[int, float] = {}
            for token, value in zip(phase.get("x_values", []), phase.get(value_key, [])):
                if isinstance(token, int) and isinstance(value, (int, float)):
                    out[int(token)] = float(value) * (1000.0 if ms else 1.0)
            return out

        return cls(
            prefill_tps=_token_map(prefill, "tps_values"),
            decode_tps=_token_map(decode, "tps_values"),
            prefill_latency_ms=_token_map(prefill, "time_values", ms=True),
            decode_duration_ms=_token_map(decode, "time_values", ms=True),
            prefill_tokens_per_j=_as_float(device.get("prefill_tok_per_j_last")),
            decode_tokens_per_j=_as_float(device.get("decode_tok_per_j_last")),
            prefill_j_per_token=_as_float(device.get("prefill_j_per_tok_last")),
            decode_j_per_token=_as_float(device.get("decode_j_per_tok_last")),
            avg_power_w=_as_float(device.get("avg_power_w")),
            p99_power_w=_as_float(device.get("p99_power_w")),
            total_energy_j=_as_float(device.get("total_energy_j")),
            avg_utilization_pct=_as_float(device.get("avg_utilization_pct")),
            p99_utilization_pct=_as_float(device.get("p99_utilization_pct")),
            avg_temperature_c=_as_float(device.get("avg_temperature_c")),
            p99_temperature_c=_as_float(device.get("p99_temperature_c")),
            avg_memory_used_mb=_as_float(device.get("avg_memory_used_mb")),
            p99_memory_used_mb=_as_float(device.get("p99_memory_used_mb")),
            avg_memory_used_pct=_as_float(device.get("avg_memory_used_pct")),
            p99_memory_used_pct=_as_float(device.get("p99_memory_used_pct")),
        )


@dataclass
class VLMCompareMetric(BaseCompareMetric):
    """Image-text-to-text compare metric."""

    llm_prefill_tps: float | None = None
    llm_decode_tps: float | None = None
    llm_ttft_ms: float | None = None
    llm_decode_duration_ms: float | None = None
    llm_total_ms: float | None = None
    vision_encode_ms: float | None = None
    vision_fps: float | None = None
    vision_img_per_j: float | None = None
    vision_j_per_img: float | None = None
    llm_prefill_tok_per_j: float | None = None
    llm_decode_tok_per_j: float | None = None
    llm_prefill_j_per_tok: float | None = None
    llm_decode_j_per_tok: float | None = None

    TASK: ClassVar[str] = "image-text-to-text"
    SCALAR_SPECS: ClassVar[Sequence[ScalarChartSpec]] = (
        ScalarChartSpec("llm_prefill_tps.png", "Prefill Tokens Per Second", "Tokens Per Second", "llm_prefill_tps"),
        ScalarChartSpec(
            "llm_prefill_tokens_per_j.png",
            "Prefill Tokens Per Joule",
            "Tokens Per Joule",
            "llm_prefill_tok_per_j",
        ),
        ScalarChartSpec("llm_decode_tps.png", "Decode Tokens Per Second", "Tokens Per Second", "llm_decode_tps"),
        ScalarChartSpec(
            "llm_decode_tokens_per_j.png",
            "Decode Tokens Per Joule",
            "Tokens Per Joule",
            "llm_decode_tok_per_j",
        ),
        ScalarChartSpec("vision_fps.png", "Vision FPS", "Frames Per Second", "vision_fps"),
        ScalarChartSpec("vision_encode_ms.png", "Vision Encode ms", "Milliseconds", "vision_encode_ms"),
        ScalarChartSpec("vision_img_per_j.png", "Vision Images Per Joule", "Images Per Joule", "vision_img_per_j"),
    )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> VLMCompareMetric | None:
        """Parse one image-text-to-text benchmark payload."""

        benchmark = payload.get("benchmark", {})
        if not isinstance(benchmark, Mapping):
            return None
        llm_results = benchmark.get("llm_results", {})
        llm_summary = llm_results.get("summary", {}) if isinstance(llm_results, Mapping) else {}
        vision_summary = benchmark.get("vision_summary", {})
        device = payload.get("device", {})
        if not isinstance(llm_summary, Mapping) or not isinstance(vision_summary, Mapping):
            return None
        if not isinstance(device, Mapping):
            device = {}
        return cls(
            llm_prefill_tps=_summary_mean(llm_summary, "llm_prefill_tps"),
            llm_decode_tps=_summary_mean(llm_summary, "llm_decode_tps"),
            llm_ttft_ms=_summary_mean(llm_summary, "llm_ttft_ms"),
            llm_decode_duration_ms=_summary_mean(llm_summary, "llm_decode_duration_ms"),
            llm_total_ms=_summary_mean(llm_summary, "llm_total_ms"),
            vision_encode_ms=_summary_mean(vision_summary, "vision_encode_ms"),
            vision_fps=_summary_mean(vision_summary, "vision_fps"),
            vision_img_per_j=_summary_mean(vision_summary, "vision_img_per_j"),
            vision_j_per_img=_summary_mean(vision_summary, "vision_j_per_img"),
            llm_prefill_tok_per_j=_summary_mean(llm_summary, "prefill_tok_per_j"),
            llm_decode_tok_per_j=_summary_mean(llm_summary, "decode_tok_per_j"),
            llm_prefill_j_per_tok=_summary_mean(llm_summary, "prefill_j_per_tok"),
            llm_decode_j_per_tok=_summary_mean(llm_summary, "decode_j_per_tok"),
            avg_power_w=_as_float(device.get("avg_power_w")),
            p99_power_w=_as_float(device.get("p99_power_w")),
            total_energy_j=_as_float(device.get("total_energy_j")),
            avg_utilization_pct=_as_float(device.get("avg_utilization_pct")),
            p99_utilization_pct=_as_float(device.get("p99_utilization_pct")),
            avg_temperature_c=_as_float(device.get("avg_temperature_c")),
            p99_temperature_c=_as_float(device.get("p99_temperature_c")),
            avg_memory_used_mb=_as_float(device.get("avg_memory_used_mb")),
            p99_memory_used_mb=_as_float(device.get("p99_memory_used_mb")),
            avg_memory_used_pct=_as_float(device.get("avg_memory_used_pct")),
            p99_memory_used_pct=_as_float(device.get("p99_memory_used_pct")),
        )


@dataclass
class ASRCompareMetric(BaseCompareMetric):
    """Automatic speech recognition compare metric."""

    wer: float | None = None
    cer: float | None = None
    rtf: float | None = None
    inverse_rtf: float | None = None
    mean_latency_s: float | None = None
    p50_latency_s: float | None = None
    p95_latency_s: float | None = None
    throughput_samples_per_s: float | None = None
    decode_tokens_per_s: float | None = None
    avg_tokens_per_sample: float | None = None

    @property
    def wer_pct(self) -> float | None:
        """Return WER in percentage units for chart display."""

        return None if self.wer is None else 100.0 * self.wer

    @property
    def cer_pct(self) -> float | None:
        """Return CER in percentage units for chart display."""

        return None if self.cer is None else 100.0 * self.cer

    TASK: ClassVar[str] = "automatic-speech-recognition"
    SCALAR_SPECS: ClassVar[Sequence[ScalarChartSpec]] = (
        ScalarChartSpec("wer.png", "Word Error Rate", "WER (%)", "wer_pct"),
        ScalarChartSpec("cer.png", "Character Error Rate", "CER (%)", "cer_pct"),
        ScalarChartSpec("rtf.png", "Real-Time Factor", "RTF", "rtf"),
        ScalarChartSpec("inverse_rtf.png", "Inverse Real-Time Factor", "x realtime", "inverse_rtf"),
        ScalarChartSpec("p95_latency_s.png", "P95 Latency", "Seconds", "p95_latency_s"),
        ScalarChartSpec(
            "throughput_samples_per_s.png",
            "Throughput",
            "Samples Per Second",
            "throughput_samples_per_s",
        ),
        ScalarChartSpec(
            "decode_tokens_per_s.png",
            "Decode Tokens Per Second",
            "Tokens Per Second",
            "decode_tokens_per_s",
        ),
    )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ASRCompareMetric | None:
        """Parse one ASR benchmark payload."""

        asr = payload.get("asr")
        if not isinstance(asr, Mapping):
            return None
        device = payload.get("device", {})
        if not isinstance(device, Mapping):
            device = {}
        return cls(
            wer=_as_float(asr.get("wer")),
            cer=_as_float(asr.get("cer")),
            rtf=_as_float(asr.get("rtf")),
            inverse_rtf=_as_float(asr.get("inverse_rtf")),
            mean_latency_s=_as_float(asr.get("mean_latency_s")),
            p50_latency_s=_as_float(asr.get("p50_latency_s")),
            p95_latency_s=_as_float(asr.get("p95_latency_s")),
            throughput_samples_per_s=_as_float(asr.get("throughput_samples_per_s")),
            decode_tokens_per_s=_as_float(asr.get("decode_tokens_per_s")),
            avg_tokens_per_sample=_as_float(asr.get("avg_tokens_per_sample")),
            avg_power_w=_as_float(device.get("avg_power_w")),
            p99_power_w=_as_float(device.get("p99_power_w")),
            total_energy_j=_as_float(device.get("total_energy_j")),
            avg_utilization_pct=_as_float(device.get("avg_utilization_pct")),
            p99_utilization_pct=_as_float(device.get("p99_utilization_pct")),
            avg_temperature_c=_as_float(device.get("avg_temperature_c")),
            p99_temperature_c=_as_float(device.get("p99_temperature_c")),
            avg_memory_used_mb=_as_float(device.get("avg_memory_used_mb")),
            p99_memory_used_mb=_as_float(device.get("p99_memory_used_mb")),
            avg_memory_used_pct=_as_float(device.get("avg_memory_used_pct")),
            p99_memory_used_pct=_as_float(device.get("p99_memory_used_pct")),
        )


TASK_REGISTRY: dict[str, type[BaseCompareMetric]] = {
    LLMCompareMetric.TASK: LLMCompareMetric,
    VLMCompareMetric.TASK: VLMCompareMetric,
    ASRCompareMetric.TASK: ASRCompareMetric,
}


def collect_metrics(folder: Path, metric_cls: type[BaseCompareMetric]) -> dict[str, BaseCompareMetric]:
    """Collect normalized per-model metrics from one results folder."""

    normalized: dict[str, BaseCompareMetric] = {}
    normalized_sources: dict[str, Path] = {}
    for path in sorted(folder.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, JSONDecodeError) as exc:
            print(f"Warning: failed to parse {path}: {exc}")
            continue
        if not isinstance(payload, Mapping):
            continue
        model_id = payload.get("model")
        if not isinstance(model_id, str) or not model_id:
            continue
        metric = metric_cls.from_payload(payload)
        if metric is None:
            continue
        norm_key = normalize_model_key(path, model_id)
        if norm_key not in normalized:
            normalized[norm_key] = metric
            normalized_sources[norm_key] = path
            continue

        original_path = normalized_sources[norm_key]
        print(
            "Warning: duplicate normalized model key "
            f"'{norm_key}' in {path}; keeping {original_path.name} and skipping {path.name}."
        )
    return normalized


def common_model_ids(metrics_by_folder: Sequence[Mapping[str, BaseCompareMetric]]) -> list[str]:
    """Return common model ids across all folders."""

    if not metrics_by_folder:
        return []
    model_sets = [set(metrics.keys()) for metrics in metrics_by_folder]
    if not model_sets:
        return []
    return sorted(set.intersection(*model_sets))


def render_charts(
    *,
    metric_cls: type[BaseCompareMetric],
    models: list[str],
    labels: list[str],
    metrics_by_folder: list[dict[str, BaseCompareMetric]],
    output_dir: Path,
) -> None:
    """Render charts declared by one metric class."""

    for spec in metric_cls.TOKEN_SPECS:
        plot_token_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            token_selector=lambda metric, attr=spec.attr: getattr(metric, attr),
            title=spec.title,
            x_label=spec.x_label,
            output_path=output_dir / spec.filename,
        )
    for spec in [*metric_cls.SCALAR_SPECS, *metric_cls.shared_scalar_specs()]:
        plot_scalar_chart(
            models=models,
            folder_labels=labels,
            metrics_by_folder=metrics_by_folder,
            scalar_selector=lambda metric, attr=spec.attr: getattr(metric, attr),
            title=spec.title,
            x_label=spec.x_label,
            output_path=output_dir / spec.filename,
        )


__all__ = [
    "ASRCompareMetric",
    "BaseCompareMetric",
    "LLMCompareMetric",
    "ScalarChartSpec",
    "TASK_REGISTRY",
    "TokenChartSpec",
    "VLMCompareMetric",
    "collect_metrics",
    "common_model_ids",
    "default_charts_dir",
    "folder_labels",
    "normalize_model_key",
    "render_charts",
]
