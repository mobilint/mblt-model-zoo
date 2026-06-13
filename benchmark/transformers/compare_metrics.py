from __future__ import annotations

import importlib.util
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

try:
    from benchmark.common.summary_utils import markdown_table
except ModuleNotFoundError:
    _summary_utils_path = Path(__file__).resolve().parents[1] / "common" / "summary_utils.py"
    _summary_spec = importlib.util.spec_from_file_location("benchmark_common_summary_utils", _summary_utils_path)
    if _summary_spec is None or _summary_spec.loader is None:
        raise
    _summary_mod = importlib.util.module_from_spec(_summary_spec)
    _summary_spec.loader.exec_module(_summary_mod)
    markdown_table = _summary_mod.markdown_table

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


def _as_float(value: Any) -> float | None:
    """Return a float for numeric values."""

    return float(value) if isinstance(value, (int, float)) else None


def _summary_mean(mapping: Mapping[str, Any], key: str) -> float | None:
    """Return ``summary[key].mean`` as a float when present."""

    value = mapping.get(key)
    if not isinstance(value, Mapping):
        return None
    return _as_float(value.get("mean"))


def _first_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    """Return the first numeric value found for one of ``keys``."""

    for key in keys:
        value = _as_float(mapping.get(key))
        if value is not None:
            return value
    return None


def payload_benchmark_type(payload: Mapping[str, Any]) -> str | None:
    """Return normalized benchmark type for compare-compatible payloads."""

    benchmark_type = payload.get("benchmark_type")
    if benchmark_type in {"measure", "sweep"}:
        return str(benchmark_type)
    benchmark = payload.get("benchmark", payload)
    if isinstance(benchmark, Mapping):
        if isinstance(benchmark.get("prefill_sweep"), Mapping) and isinstance(benchmark.get("decode_sweep"), Mapping):
            return "sweep"
        if isinstance(benchmark.get("llm_results"), Mapping) and isinstance(benchmark.get("vision_summary"), Mapping):
            return "sweep"
    return None


def _strip_group_id(model_id: str) -> str:
    """Compare by model id only, ignoring a leading group id prefix."""

    return model_id.split("__", 1)[1] if "__" in model_id else model_id


def _restore_safe_model_id(value: str) -> str:
    """Restore slash-separated model ids from benchmark-safe filenames when possible."""

    if "/" in value:
        return _strip_group_id(value)
    if "__" in value:
        return value.replace("__", "/", 1)
    return value


def _model_name(model_id: str) -> str:
    """Return the repository name without a leading Hugging Face owner id."""

    restored = _restore_safe_model_id(model_id)
    return restored.rsplit("/", 1)[1] if "/" in restored else restored


def payload_task(payload: Mapping[str, Any]) -> str | None:
    """Return the benchmark task from normalized and legacy payload schemas."""

    task = payload.get("task")
    if isinstance(task, str) and task:
        return task

    benchmark_type = payload.get("benchmark_type")
    if isinstance(benchmark_type, str) and benchmark_type in TASK_REGISTRY:
        return benchmark_type
    return None


def _compare_model_id(model_id: str, *, strip_owner: bool = False) -> str:
    """Return a restored model id, optionally ignoring a leading owner id."""

    restored = _restore_safe_model_id(model_id)
    if strip_owner:
        return _model_name(restored)
    return restored


def normalize_model_key(path: Path, loaded_model_id: str, *, strip_owner: bool = False) -> str:
    """Normalize a model id for cross-folder comparison."""

    stem = path.stem
    if "_beams" in stem:
        restored_stem = _restore_safe_model_id(stem)
        if loaded_model_id.endswith(stem) or loaded_model_id.endswith(restored_stem):
            return _compare_model_id(loaded_model_id, strip_owner=strip_owner)
        beam_suffix = stem.rsplit("_beams", 1)[1]
        return f"{_compare_model_id(loaded_model_id, strip_owner=strip_owner)}_beams{beam_suffix}"
    if "__" in stem:
        return _compare_model_id(stem, strip_owner=strip_owner)
    return _compare_model_id(loaded_model_id, strip_owner=strip_owner)


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

        device = payload.get("device", {})
        if not isinstance(device, Mapping):
            device = {}
        if payload_benchmark_type(payload) == "measure":
            summary = payload.get("summary", {})
            if not isinstance(summary, Mapping):
                return None
            prefill = payload.get("prefill")
            decode = payload.get("decode")
            prefill_token = int(prefill) if isinstance(prefill, int) else None
            decode_token = int(decode) if isinstance(decode, int) else None
            prefill_tps = _summary_mean(summary, "prefill_tps")
            decode_tps = _summary_mean(summary, "decode_tps")
            prefill_latency_ms = _summary_mean(summary, "ttft_ms")
            decode_duration_ms = _summary_mean(summary, "decode_duration_ms")
            return cls(
                prefill_tps={prefill_token: prefill_tps} if prefill_token is not None and prefill_tps is not None else {},
                decode_tps={decode_token: decode_tps} if decode_token is not None and decode_tps is not None else {},
                prefill_latency_ms=(
                    {prefill_token: prefill_latency_ms}
                    if prefill_token is not None and prefill_latency_ms is not None
                    else {}
                ),
                decode_duration_ms=(
                    {decode_token: decode_duration_ms}
                    if decode_token is not None and decode_duration_ms is not None
                    else {}
                ),
                prefill_tokens_per_j=_first_float(device, "prefill_tokens_per_j", "prefill_tok_per_j_last"),
                decode_tokens_per_j=_first_float(device, "decode_tokens_per_j", "decode_tok_per_j_last"),
                prefill_j_per_token=_first_float(device, "prefill_j_per_token", "prefill_j_per_tok_last"),
                decode_j_per_token=_first_float(device, "decode_j_per_token", "decode_j_per_tok_last"),
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

        benchmark = payload.get("benchmark", payload)
        if not isinstance(benchmark, Mapping):
            return None
        prefill = benchmark.get("prefill_sweep", {})
        decode = benchmark.get("decode_sweep", {})
        if not isinstance(prefill, Mapping) or not isinstance(decode, Mapping):
            return None
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

        device = payload.get("device", {})
        if not isinstance(device, Mapping):
            device = {}
        if payload_benchmark_type(payload) == "measure":
            summary = payload.get("summary", {})
            if not isinstance(summary, Mapping):
                return None
            return cls(
                llm_prefill_tps=_summary_mean(summary, "llm_prefill_tps"),
                llm_decode_tps=_summary_mean(summary, "llm_decode_tps"),
                llm_ttft_ms=_summary_mean(summary, "llm_ttft_ms"),
                llm_decode_duration_ms=_summary_mean(summary, "llm_decode_duration_ms"),
                vision_encode_ms=_summary_mean(summary, "vision_encode_ms"),
                vision_fps=_summary_mean(summary, "vision_fps"),
                vision_img_per_j=_as_float(device.get("vision_img_per_j")),
                llm_prefill_tok_per_j=_as_float(device.get("llm_prefill_tok_per_j")),
                llm_decode_tok_per_j=_as_float(device.get("llm_decode_tok_per_j")),
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

        benchmark = payload.get("benchmark", {})
        if not isinstance(benchmark, Mapping):
            return None
        llm_results = benchmark.get("llm_results", {})
        llm_summary = llm_results.get("summary", {}) if isinstance(llm_results, Mapping) else {}
        vision_summary = benchmark.get("vision_summary", {})
        if not isinstance(llm_summary, Mapping) or not isinstance(vision_summary, Mapping):
            return None
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
    sec_per_j: float | None = None
    j_per_sec: float | None = None
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
        ScalarChartSpec("sec_per_j.png", "Seconds Per Joule", "Seconds Per Joule", "sec_per_j"),
        ScalarChartSpec("j_per_sec.png", "Joules Per Audio Second", "Joules Per Audio Second", "j_per_sec"),
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
            sec_per_j=_as_float(device.get("sec_per_j")),
            j_per_sec=_as_float(device.get("j_per_sec")),
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


def collect_metrics(
    folder: Path,
    metric_cls: type[BaseCompareMetric],
    *,
    benchmark_type: str | None = None,
    strip_owner: bool = False,
) -> dict[str, BaseCompareMetric]:
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
        detected_benchmark_type = payload_benchmark_type(payload)
        if benchmark_type is not None and detected_benchmark_type is not None and detected_benchmark_type != benchmark_type:
            print(
                f"Warning: skipping {path.name} because benchmark_type '{detected_benchmark_type}' "
                f"does not match requested benchmark_type '{benchmark_type}'."
            )
            continue
        detected_task = payload_task(payload)
        if detected_task is None and payload.get("benchmark_type") == "measure" and "task" not in payload:
            print(
                f"Warning: {path.name} has benchmark_type='measure' but no task field; "
                f"attempting to parse as '{metric_cls.TASK}' for legacy compatibility."
            )
        if detected_task is not None and detected_task != metric_cls.TASK:
            print(
                f"Warning: skipping {path.name} because task '{detected_task}' "
                f"does not match requested task '{metric_cls.TASK}'."
            )
            continue
        model_id = payload.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            model_id = payload.get("model")
        if not isinstance(model_id, str) or not model_id:
            continue
        metric = metric_cls.from_payload(payload)
        if metric is None:
            status = payload.get("status")
            if isinstance(status, str) and status:
                reason = payload.get("reason", "")
                print(
                    f"Warning: skipping status-only payload {path} "
                    f"(status={status}, reason={reason})."
                )
            continue
        norm_key = normalize_model_key(path, model_id, strip_owner=strip_owner)
        if norm_key not in normalized:
            normalized[norm_key] = metric
            normalized_sources[norm_key] = path
            continue

        original_path = normalized_sources[norm_key]
        strip_hint = " with --strip-owner" if strip_owner else ""
        print(
            "Warning: duplicate normalized model key "
            f"'{norm_key}'{strip_hint}; keeping source '{original_path}' and skipping duplicate source '{path}'."
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


def build_compare_plot_tables(
    *,
    metric_cls: type[BaseCompareMetric],
    models: Sequence[str],
    labels: Sequence[str],
    metrics_by_folder: Sequence[Mapping[str, BaseCompareMetric]],
) -> dict[str, str]:
    """Build plot-specific Markdown tables for benchmark comparison summaries."""

    tables: dict[str, str] = {}
    for spec in metric_cls.TOKEN_SPECS:
        table = _token_compare_table(
            models=models,
            labels=labels,
            metrics_by_folder=metrics_by_folder,
            attr=spec.attr,
        )
        if table:
            tables[spec.filename] = table
    for spec in [*metric_cls.SCALAR_SPECS, *metric_cls.shared_scalar_specs()]:
        table = _scalar_compare_table(
            models=models,
            labels=labels,
            metrics_by_folder=metrics_by_folder,
            attr=spec.attr,
            unit_header=spec.x_label,
        )
        if table:
            tables[spec.filename] = table
    return tables


def write_compare_markdown(
    path: Path | str,
    *,
    metric_cls: type[BaseCompareMetric],
    models: Sequence[str],
    labels: Sequence[str],
    metrics_by_folder: Sequence[Mapping[str, BaseCompareMetric]],
) -> None:
    """Write a combined Markdown table for benchmark comparison metrics."""

    Path(path).write_text(
        build_compare_markdown(
            metric_cls=metric_cls,
            models=models,
            labels=labels,
            metrics_by_folder=metrics_by_folder,
        ),
        encoding="utf-8",
    )


def build_compare_markdown(
    *,
    metric_cls: type[BaseCompareMetric],
    models: Sequence[str],
    labels: Sequence[str],
    metrics_by_folder: Sequence[Mapping[str, BaseCompareMetric]],
) -> str:
    """Build one wide Markdown table containing all comparable task metrics."""

    headers: list[str] = ["Model"]
    value_getters: list[tuple[int, str, int | None]] = []

    for spec in metric_cls.TOKEN_SPECS:
        for token in _tokens_for_attr(models=models, metrics_by_folder=metrics_by_folder, attr=spec.attr):
            for folder_idx, label in enumerate(labels):
                headers.append(f"{label} {spec.title} ({token} tokens)")
                value_getters.append((folder_idx, spec.attr, token))
    for spec in [*metric_cls.SCALAR_SPECS, *metric_cls.shared_scalar_specs()]:
        for folder_idx, label in enumerate(labels):
            headers.append(f"{label} {spec.title} ({spec.x_label})")
            value_getters.append((folder_idx, spec.attr, None))

    rows: list[list[Any]] = []
    for model in models:
        row: list[Any] = [model]
        for folder_idx, attr, token in value_getters:
            metric = metrics_by_folder[folder_idx][model]
            row.append(_metric_value(metric, attr, token=token))
        rows.append(row)
    return markdown_table(headers, rows) if rows else "No common benchmark results found.\n"


def _token_compare_table(
    *,
    models: Sequence[str],
    labels: Sequence[str],
    metrics_by_folder: Sequence[Mapping[str, BaseCompareMetric]],
    attr: str,
) -> str:
    """Build a Markdown table for one token-keyed comparison plot."""

    tokens = _tokens_for_attr(models=models, metrics_by_folder=metrics_by_folder, attr=attr)
    if not tokens:
        return ""
    headers = ["Model", *(f"{label} {token} tokens" for token in tokens for label in labels)]
    rows = [
        [
            model,
            *(
                _metric_value(metrics_by_folder[folder_idx][model], attr, token=token)
                for token in tokens
                for folder_idx in range(len(labels))
            ),
        ]
        for model in models
    ]
    return markdown_table(headers, rows)


def _scalar_compare_table(
    *,
    models: Sequence[str],
    labels: Sequence[str],
    metrics_by_folder: Sequence[Mapping[str, BaseCompareMetric]],
    attr: str,
    unit_header: str,
) -> str:
    """Build a Markdown table for one scalar comparison plot."""

    headers = ["Model", *(f"{label} {unit_header}" for label in labels)]
    rows = [
        [model, *(_metric_value(metrics_by_folder[folder_idx][model], attr) for folder_idx in range(len(labels)))]
        for model in models
    ]
    return markdown_table(headers, rows)


def _tokens_for_attr(
    *,
    models: Sequence[str],
    metrics_by_folder: Sequence[Mapping[str, BaseCompareMetric]],
    attr: str,
) -> list[int]:
    """Return sorted token keys available for one metric attribute."""

    tokens: set[int] = set()
    for model in models:
        for folder_metrics in metrics_by_folder:
            values = getattr(folder_metrics[model], attr)
            if isinstance(values, Mapping):
                tokens.update(int(token) for token in values if isinstance(token, int))
    return sorted(tokens)


def _metric_value(metric: BaseCompareMetric, attr: str, *, token: int | None = None) -> Any:
    """Return a scalar or token-keyed metric value."""

    value = getattr(metric, attr)
    if token is None:
        return value
    return value.get(token) if isinstance(value, Mapping) else None


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
    "build_compare_markdown",
    "build_compare_plot_tables",
    "default_charts_dir",
    "folder_labels",
    "normalize_model_key",
    "payload_benchmark_type",
    "payload_task",
    "render_charts",
    "write_compare_markdown",
]
