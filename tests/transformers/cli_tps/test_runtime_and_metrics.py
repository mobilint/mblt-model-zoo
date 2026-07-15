import argparse
import importlib
import inspect
import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from transformers import GenerationConfig, GenerationMixin

from mblt_model_zoo.cli import tps as tps_cli
from mblt_model_zoo.cli.main import build_parser
from mblt_model_zoo.hf_transformers.models.blip.modeling_blip_text import MobilintBlipTextLMHeadModel
from mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl import MobilintQwen2VLForConditionalGeneration
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    DEVICE_METRIC_KEYS,
    build_device_tracker,
    extract_device_metric,
    extract_device_time_series,
    integrate_power_trace_j,
    parse_npu_rail_metrics,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
    BenchmarkResult,
    SingleMeasurement,
    SweepData,
    TPSMeasurer,
    VLMTPSMeasurer,
    _get_npu_timing_target,
    _resolve_config_vocab_size,
    _resolve_image_features_tensor,
    _supports_fake_decode_prefill,
    _temporarily_sanitize_generation_config,
    npu_latency_pct,
)
from mblt_model_zoo.hf_transformers.utils.generation_utils import MobilintGenerationMixin
from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin


class _DummyConfig:
    def __init__(self, vocab_size: int | None = None, text_config=None, max_batch_size=None, vision_config=None):
        if vocab_size is not None:
            self.vocab_size = vocab_size
        if text_config is not None:
            self.text_config = text_config
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        if vision_config is not None:
            self.vision_config = vision_config


class _DummyVisionOutput:
    def __init__(self, pooler_output=None):
        self.pooler_output = pooler_output


class _DummyMxqModel:
    """Minimal cache MXQ marker for fake prefill routing tests."""


class _DummyNPUModel:
    """NPU-backed model marker exposing a cache MXQ model."""

    npu_backend = object()

    def __init__(self) -> None:
        self.mxq_model = _DummyMxqModel()

    def get_cache_mxq_model(self) -> _DummyMxqModel:
        """Return a fake MXQ model for MobilintCache construction."""
        return self.mxq_model


class _DummyTokenizer:
    """Minimal tokenizer object for text TPS streamer construction."""

    eos_token_id = 0

    def decode(self, token_ids, **kwargs) -> str:
        """Return a deterministic decoded token string."""
        del token_ids, kwargs
        return "token"


class _DummyGenerateNPUModel(_DummyNPUModel):
    """NPU-backed model double that records generate kwargs."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _DummyConfig(vocab_size=128)
        self.generation_config = GenerationConfig()
        self.generation_config.temperature = 0.6
        self.generation_config.top_p = 0.95
        self.generation_config.top_k = 20
        self.device = torch.device("cpu")
        self.generate_kwargs = None
        self.generation_config_during_generate = None

    def eval(self) -> "_DummyGenerateNPUModel":
        """Match the minimal model API used by TPSMeasurer."""
        return self

    def generate(self, **kwargs) -> torch.Tensor:
        """Record generation kwargs and emit deterministic generated tokens."""
        self.generate_kwargs = kwargs
        self.generation_config_during_generate = {
            "do_sample": self.generation_config.do_sample,
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
            "top_k": self.generation_config.top_k,
        }
        input_ids = kwargs["input_ids"]
        new_tokens = int(kwargs["max_new_tokens"])
        outputs = torch.zeros((int(input_ids.shape[0]), int(input_ids.shape[1]) + new_tokens), dtype=torch.long)
        streamer = kwargs.get("streamer")
        if streamer is not None:
            streamer.put(input_ids)
        stopping_criteria = kwargs.get("stopping_criteria")
        if stopping_criteria is not None:
            stopping_criteria(outputs[:, : int(input_ids.shape[1]) + 1], None)
        if streamer is not None:
            for _ in range(new_tokens):
                streamer.put(torch.zeros((int(input_ids.shape[0]), 1), dtype=torch.long))
            streamer.end()
        return outputs


class _DummyBatchedGenerateNPUModel(_DummyGenerateNPUModel):
    """NPU-backed model double for batched generate callback tests."""

    def generate(self, **kwargs) -> torch.Tensor:
        """Return batched generated ids without requiring a streamer."""
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        new_tokens = int(kwargs["max_new_tokens"])
        outputs = torch.zeros((int(input_ids.shape[0]), int(input_ids.shape[1]) + new_tokens), dtype=torch.long)
        stopping_criteria = kwargs.get("stopping_criteria")
        if stopping_criteria is not None:
            stopping_criteria(outputs[:, : int(input_ids.shape[1]) + 1], None)
        return outputs


class _DummyTextPipeline:
    """Minimal text-generation pipeline for TPSMeasurer tests."""

    def __init__(self, model: _DummyGenerateNPUModel) -> None:
        self.model = model
        self.tokenizer = _DummyTokenizer()


class _DummyEagle3AcceptanceModel:
    """Model double exposing EAGLE-3 acceptance stats through the public property."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.last_eagle3_acceptance_stats = {
            "steps": 3,
            "accepted_tokens_sum": 7,
            "accepted_tokens_avg": 7.0 / 3.0,
            "acceptance_ratio": 0.5,
        }

    def eval(self) -> "_DummyEagle3AcceptanceModel":
        """Match the minimal model API used by TPSMeasurer."""
        return self


class _DummyNonNPUModel:
    """Non-NPU model marker without Mobilint cache support."""


class _DummyVLMContainer:
    """VLM-like container with a nested language model."""

    def __init__(self, language_model) -> None:
        self.model = type("NestedModel", (), {"language_model": language_model})()


class _DummyNestedTimingModel:
    """VLM-like model exposing NPU timing only through the nested language model."""

    def __init__(self) -> None:
        self.model = type("NestedModel", (), {"language_model": _DummyNPUModel()})()


class _DummyAggregateTimingModel:
    """Model without an npu_backend marker that still exposes aggregate timing APIs."""

    def reset_npu_timing(self) -> None:
        """Match Mobilint aggregate timing reset API."""

    def get_npu_timing(self) -> dict[str, float | int]:
        """Return a non-empty aggregate timing payload."""
        return {"prefill_time": 0.1, "decode_time": 0.2, "prefill_calls": 1, "decode_calls": 2}


class _DummyFakePrefillCache:
    """Minimal cache object used by VLM fake-prefill decode tests."""

    def __init__(self) -> None:
        self.prefill_length: int | None = None

    def fake_prefill(self, sequence_length: int) -> None:
        """Record the fake prefill length requested by the benchmark."""
        self.prefill_length = sequence_length


class _DummyVLMDecodeOutput:
    """Minimal language-model output carrying logits-like hidden states."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _DummyVLMLanguageModel(torch.nn.Module):
    """Tiny language model double for fake-prefill decode accounting tests."""

    npu_backend = object()

    def __init__(self, vocab_size: int = 16, hidden_size: int = 4) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.cache = _DummyFakePrefillCache()
        self.calls = 0
        self.npu_time: float | None = None

    def get_cache_mxq_model(self) -> _DummyMxqModel:
        """Return a fake MXQ marker so the benchmark enables fake-prefill decode."""
        return _DummyMxqModel()

    def get_input_embeddings(self) -> torch.nn.Embedding:
        """Return token embeddings used by the decode loop."""
        return self.embedding

    def _get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int) -> _DummyFakePrefillCache:
        """Return a fake cache object compatible with the benchmark helper."""
        del cache_implementation, batch_size, max_cache_len
        return self.cache

    def forward(self, **kwargs) -> _DummyVLMDecodeOutput:
        """Simulate one decode step and expose per-step NPU timing."""
        del kwargs
        self.calls += 1
        self.npu_time = 0.25
        vocab_size = self.embedding.num_embeddings
        logits = torch.zeros((1, 1, vocab_size), dtype=torch.float32)
        logits[0, 0, self.calls % vocab_size] = 1.0
        return _DummyVLMDecodeOutput(logits)


class _DummyVLMModel:
    """Minimal VLM container exposing config and device for the benchmark."""

    def __init__(self, language_model: _DummyVLMLanguageModel) -> None:
        self.device = torch.device("cpu")
        self.config = _DummyConfig(vocab_size=language_model.embedding.num_embeddings)
        self.language_model = language_model
        self.model = type("NestedModel", (), {"language_model": language_model})()


class _DummyBatchedVLMModel:
    """Minimal batched VLM generation model returning prompt plus generated tokens."""

    def __init__(self, language_model: _DummyVLMLanguageModel, generated_tokens: int) -> None:
        self.device = torch.device("cpu")
        self.config = _DummyConfig(vocab_size=language_model.embedding.num_embeddings)
        self.language_model = language_model
        self.model = type("NestedModel", (), {"language_model": language_model})()
        self.generated_tokens = generated_tokens

    def generate(self, **kwargs) -> torch.Tensor:
        """Return token ids whose length includes the prompt prefix."""
        inputs_embeds = kwargs["inputs_embeds"]
        batch_size, seq_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        return torch.zeros((batch_size, seq_len + self.generated_tokens), dtype=torch.long)


class _DummyVLMTPSMeasurer(VLMTPSMeasurer):
    """VLM TPS measurer double that bypasses pipeline construction."""

    def __init__(self, language_model: _DummyVLMLanguageModel) -> None:
        self.model = _DummyVLMModel(language_model)
        self.tokenizer = object()
        self.processor = object()

    def _get_language_model(self) -> _DummyVLMLanguageModel:
        """Return the dummy language model under test."""
        return self.model.language_model


class _DummyBatchedVLMTPSMeasurer(_DummyVLMTPSMeasurer):
    """VLM TPS measurer double for batched generation accounting tests."""

    def __init__(self, language_model: _DummyVLMLanguageModel, generated_tokens: int) -> None:
        self.model = _DummyBatchedVLMModel(language_model, generated_tokens)
        self.tokenizer = _DummyTokenizer()
        self.processor = object()


class _DummyVisionLatencyVLMTPSMeasurer(_DummyVLMTPSMeasurer):
    """VLM TPS measurer double with deterministic batch vision latency."""

    def __init__(self, language_model: _DummyVLMLanguageModel, batch_latency: float) -> None:
        super().__init__(language_model)
        self.batch_latency = batch_latency

    def _build_inputs(self, image_resolution: int, prompt: str, batch_size: int = 1) -> dict[str, object]:
        """Return minimal inputs that preserve batch metadata for tests."""
        return {"image_resolution": image_resolution, "prompt": prompt, "batch_size": batch_size}

    def _measure_vision_encode(self, inputs: dict) -> tuple[float, torch.Tensor]:
        """Return a deterministic latency for the whole vision batch."""
        del inputs
        return self.batch_latency, torch.zeros((1, 4), dtype=torch.float32)

    def _build_inputs_embeds(self, inputs: dict, image_features: torch.Tensor) -> torch.Tensor:
        """Return dummy embeddings for the requested batch size."""
        del image_features
        return torch.zeros((int(inputs["batch_size"]), 2, 4), dtype=torch.float32)

    def _measure_llm_once(
        self,
        inputs_embeds: torch.Tensor,
        num_decode: int,
        npu_prefill_chunk_size=None,
        show_progress: bool = False,
        progress_desc=None,
    ) -> SingleMeasurement:
        """Return a lightweight LLM measurement without running generation."""
        del inputs_embeds, npu_prefill_chunk_size, show_progress, progress_desc
        return SingleMeasurement(
            num_prefill=2,
            num_decode=int(num_decode),
            prefill_latency=1.0,
            prefill_tps=2.0,
            decode_duration=1.0,
            decode_tps=float(num_decode),
            total_time=2.0,
            avg_total_prefill_token_latency=0.5,
            avg_npu_prefill_token_latency=None,
            avg_total_decode_token_latency=1.0 / int(num_decode),
            avg_npu_decode_token_latency=None,
        )


class _RoutingTPSMeasurer(TPSMeasurer):
    """TPS measurer test double that records real/fake decode routing."""

    def __init__(self, model) -> None:
        self.model = model
        self.calls: list[tuple[str, int, int]] = []

    def measure(
        self,
        num_prefill=512,
        num_decode=128,
        npu_prefill_chunk_size=None,
        trace_path=None,
        show_progress: bool = False,
        progress_desc=None,
        on_prefill_start=None,
        on_prefill_end=None,
        on_decode_start=None,
        on_decode_end=None,
        batch_size: int = 1,
    ) -> SingleMeasurement:
        """Record real-prefill measurements without running generation."""
        del batch_size
        self.calls.append(("real", int(num_prefill), int(num_decode)))
        return SingleMeasurement(
            num_prefill=int(num_prefill),
            num_decode=int(num_decode),
            prefill_latency=1.0,
            prefill_tps=float(num_prefill),
            decode_duration=1.0,
            decode_tps=float(num_decode),
            total_time=2.0,
            avg_total_prefill_token_latency=1.0 / int(num_prefill),
            avg_npu_prefill_token_latency=None,
            avg_total_decode_token_latency=1.0 / int(num_decode),
            avg_npu_decode_token_latency=None,
        )

    def measure_decode_with_fake_prefill(
        self,
        cache_len: int,
        num_decode: int,
        trace_path=None,
        show_progress: bool = False,
        progress_desc=None,
        batch_size: int = 1,
    ) -> SingleMeasurement:
        """Record fake-prefill decode measurements without running generation."""
        del batch_size
        self.calls.append(("fake", int(cache_len), int(num_decode)))
        return SingleMeasurement(
            num_prefill=int(cache_len),
            num_decode=int(num_decode),
            prefill_latency=0.0,
            prefill_tps=0.0,
            decode_duration=1.0,
            decode_tps=float(num_decode),
            total_time=1.0,
            avg_total_prefill_token_latency=0.0,
            avg_npu_prefill_token_latency=None,
            avg_total_decode_token_latency=1.0 / int(num_decode),
            avg_npu_decode_token_latency=None,
            decode_prefill_mode="fake",
        )




def test_enrich_single_run_device_uses_batched_token_count_for_energy_metrics():
    """Energy efficiency metrics should use total tokens across the batch."""
    measurement = SingleMeasurement(
        num_prefill=4,
        num_decode=2,
        prefill_latency=2.0,
        prefill_tps=8.0,
        decode_duration=1.0,
        decode_tps=8.0,
        total_time=3.0,
        avg_total_prefill_token_latency=0.25,
        avg_npu_prefill_token_latency=None,
        avg_total_decode_token_latency=0.25,
        avg_npu_decode_token_latency=None,
    )

    tps_cli._enrich_single_run_device(
        run=measurement,
        prefill_metric={"avg_power_w": 2.0},
        decode_metric={"avg_power_w": 4.0},
        batch_size=3,
        prefill_time_series={"power_w": [{"timestamp_s": 0.0, "value": 2.0}, {"timestamp_s": 2.0, "value": 2.0}]},
        decode_time_series={"power_w": [{"timestamp_s": 0.0, "value": 4.0}, {"timestamp_s": 1.0, "value": 4.0}]},
    )

    assert measurement.prefill_tps_per_w == pytest.approx(3.0)
    assert measurement.prefill_j_per_token == pytest.approx(1.0 / 3.0)
    assert measurement.decode_tps_per_w == pytest.approx(1.5)
    assert measurement.decode_j_per_token == pytest.approx(2.0 / 3.0)
    assert measurement.total_tps_per_w == pytest.approx(2.25)
    assert measurement.total_j_per_token == pytest.approx(4.0 / 9.0)


def test_enrich_single_run_device_computes_energy_metrics_without_avg_power():
    """Trace-integrated energy metrics should not require scalar average power values."""
    measurement = SingleMeasurement(
        num_prefill=4,
        num_decode=2,
        prefill_latency=2.0,
        prefill_tps=8.0,
        decode_duration=1.0,
        decode_tps=8.0,
        total_time=3.0,
        avg_total_prefill_token_latency=0.25,
        avg_npu_prefill_token_latency=None,
        avg_total_decode_token_latency=0.25,
        avg_npu_decode_token_latency=None,
    )

    tps_cli._enrich_single_run_device(
        run=measurement,
        prefill_metric={},
        decode_metric={},
        batch_size=2,
        prefill_time_series={"power_w": [{"timestamp_s": 0.0, "value": 2.0}, {"timestamp_s": 2.0, "value": 2.0}]},
        decode_time_series={"power_w": [{"timestamp_s": 0.0, "value": 4.0}, {"timestamp_s": 1.0, "value": 4.0}]},
    )

    assert measurement.avg_power_w is None
    assert measurement.total_energy_j == pytest.approx(8.0)
    assert measurement.prefill_tps_per_w == pytest.approx(2.0)
    assert measurement.decode_tps_per_w == pytest.approx(1.0)
    assert measurement.total_tps_per_w == pytest.approx(1.5)


def test_integrate_power_trace_j_uses_trapezoidal_rule():
    """Power traces should be integrated from time-series samples, not average power fallbacks."""
    trace = [
        {"timestamp_s": 2.0, "value": 4.0},
        {"timestamp_s": 0.0, "value": 2.0},
        {"timestamp_s": 1.0, "value": 6.0},
    ]

    assert integrate_power_trace_j(trace) == pytest.approx(9.0)


def test_integrate_power_trace_j_requires_two_valid_points():
    """A single trace sample cannot produce a reliable energy integration."""
    assert integrate_power_trace_j([{"timestamp_s": 0.0, "value": 2.0}]) is None


def test_run_text_measure_forwards_resolved_batch_size(monkeypatch):
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    calls: list[int] = []
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=4)))

    class _FakeTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure(self, **kwargs) -> SingleMeasurement:
            calls.append(kwargs["batch_size"])
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

    captured_pipeline_kwargs: dict[str, object] = {}

    def _fake_build_pipeline(**kwargs):
        captured_pipeline_kwargs.update(kwargs)
        return pipeline

    monkeypatch.setattr(tps_cli, "_build_pipeline", _fake_build_pipeline)
    monkeypatch.setattr(tps_cli, "_build_phase_trackers", lambda args, pipeline: (None, None))
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "TPSMeasurer", _FakeTPSMeasurer)

    args = argparse.Namespace(
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
        batch_size=None,
        warmup=1,
        repeat=1,
        prefill=8,
        decode=2,
        npu_prefill_chunk_size=None,
        trace=None,
        device_metrics=False,
        json=None,
        device_backend="none",
    )

    assert tps_cli._run_text_measure(args) == 0
    assert calls == [4, 4]
    assert isinstance(captured_pipeline_kwargs.get("eagle3_options"), tps_cli.Eagle3PipelineOptions)


def test_run_text_sweep_forwards_resolved_batch_size(monkeypatch):
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    measure_calls: list[tuple[int, int, int]] = []
    full_calls: list[int] = []
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=3)))

    class _FakeTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure(self, **kwargs) -> SingleMeasurement:
            measure_calls.append((kwargs["batch_size"], kwargs["num_prefill"], kwargs["num_decode"]))
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
            full_calls.append(kwargs["batch_size"])
            result = BenchmarkResult()
            result.prefill_sweep.x_values.append(8)
            result.prefill_sweep.tps_values.append(1.0)
            result.prefill_sweep.time_values.append(1.0)
            result.prefill_sweep.avg_total_token_latency_values.append(1.0)
            result.prefill_sweep.avg_npu_token_latency_values.append(None)
            result.decode_sweep.x_values.append(4)
            result.decode_sweep.tps_values.append(1.0)
            result.decode_sweep.time_values.append(1.0)
            result.decode_sweep.avg_total_token_latency_values.append(1.0)
            result.decode_sweep.avg_npu_token_latency_values.append(None)
            return result

        def plot_and_save(self, result, save_path) -> None:
            del result, save_path

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_phase_trackers", lambda args, pipeline: (None, None))
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "TPSMeasurer", _FakeTPSMeasurer)

    args = argparse.Namespace(
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
        batch_size=None,
        warmup=1,
        repeat=1,
        prefill_range=(8, 8, 1),
        cache_lengths=[4],
        decode_window=2,
        npu_prefill_chunk_size=None,
        trace=None,
        device_metrics=False,
        json=None,
        csv=None,
        plot=None,
        device_backend="none",
    )

    assert tps_cli._run_text_sweep(args) == 0
    assert measure_calls == [(3, 128, 32)]
    assert full_calls == [3]


def test_run_text_sweep_repeat_aggregates_trace_energy_scope(monkeypatch, tmp_path):
    """Verify text sweep repeat energy and TPS/W use the repeated sweep scope."""
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    tracker_pairs: list[tuple[str, str]] = []
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=2)))

    class _FakeTracker:
        def __init__(self, name: str, power_w: float) -> None:
            self.name = name
            self.power_w = power_w

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def get_metric(self) -> dict[str, float]:
            return {"avg_power_w": self.power_w, "p99_power_w": self.power_w}

        def get_total_power_trace(self) -> list[tuple[float, float]]:
            return [(0.0, self.power_w), (1.0, self.power_w)]

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
            if kwargs.get("on_prefill_start") is not None:
                kwargs["on_prefill_start"]()
                kwargs["on_prefill_end"]()
                kwargs["on_decode_start"]()
                kwargs["on_decode_end"]()
            result = BenchmarkResult()
            result.prefill_sweep.x_values.extend([8, 16])
            result.prefill_sweep.tps_values.extend([8.0, 16.0])
            result.prefill_sweep.time_values.extend([1.0, 1.0])
            result.prefill_sweep.avg_total_token_latency_values.extend([0.125, 0.0625])
            result.prefill_sweep.avg_npu_token_latency_values.extend([None, None])
            result.decode_sweep.x_values.extend([4, 8])
            result.decode_sweep.tps_values.extend([4.0, 8.0])
            result.decode_sweep.time_values.extend([1.0, 1.0])
            result.decode_sweep.avg_total_token_latency_values.extend([0.5, 0.5])
            result.decode_sweep.avg_npu_token_latency_values.extend([None, None])
            return result

        def plot_and_save(self, result, save_path) -> None:
            del result, save_path

    tracker_specs = iter(
        [
            ("status-prefill", 0.0, "status-decode", 0.0),
            ("run1-prefill", 2.0, "run1-decode", 3.0),
            ("run2-prefill", 4.0, "run2-decode", 6.0),
        ]
    )

    def _fake_build_phase_trackers(args, pipeline):
        del args, pipeline
        prefill_name, prefill_power, decode_name, decode_power = next(tracker_specs)
        tracker_pairs.append((prefill_name, decode_name))
        return _FakeTracker(prefill_name, prefill_power), _FakeTracker(decode_name, decode_power)

    json_path = tmp_path / "sweep.json"
    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_phase_trackers", _fake_build_phase_trackers)
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "TPSMeasurer", _FakeTPSMeasurer)

    args = argparse.Namespace(
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
        batch_size=None,
        warmup=0,
        repeat=2,
        prefill_range=(8, 16, 8),
        cache_lengths=[4, 8],
        decode_window=2,
        npu_prefill_chunk_size=None,
        trace=None,
        device_metrics=True,
        json=str(json_path),
        csv=None,
        plot=None,
        device_backend="npu",
    )

    assert tps_cli._run_text_sweep(args) == 0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    runs = payload["runs"]

    assert tracker_pairs == [
        ("status-prefill", "status-decode"),
        ("run1-prefill", "run1-decode"),
        ("run2-prefill", "run2-decode"),
    ]
    assert [run["prefill_energy_j"] for run in runs] == [pytest.approx(2.0), pytest.approx(4.0)]
    assert [run["decode_energy_j"] for run in runs] == [pytest.approx(3.0), pytest.approx(6.0)]
    assert [run["total_energy_j"] for run in runs] == [pytest.approx(5.0), pytest.approx(10.0)]
    assert runs[0]["prefill_tps_per_w"] == pytest.approx(((8 + 16) * 2) / 2.0)
    assert runs[0]["decode_tps_per_w"] == pytest.approx((2 * 2 * 2) / 3.0)

    assert aggregate["prefill_energy_j"] == pytest.approx(6.0)
    assert aggregate["decode_energy_j"] == pytest.approx(9.0)
    assert aggregate["total_energy_j"] == pytest.approx(15.0)
    assert aggregate["prefill_tps_per_w"] == pytest.approx(((8 + 16) * 2 * 2) / 6.0)
    assert aggregate["decode_tps_per_w"] == pytest.approx((2 * 2 * 2 * 2) / 9.0)
    assert aggregate["total_tps_per_w"] == pytest.approx((((8 + 16) * 2 * 2) + (2 * 2 * 2 * 2)) / 15.0)
    assert len(payload["device_time_series_runs"]) == 2


def test_run_vlm_sweep_writes_plot(monkeypatch, tmp_path):
    """Verify VLM sweeps honor the shared --plot output path."""
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=1)))

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure_vision(self, **kwargs) -> list[tuple[float, float]]:
            return [(float(kwargs["image_resolution"]) / 1000.0, 10.0)]

        def measure_llm_full(self, **kwargs) -> BenchmarkResult:
            del kwargs
            result = BenchmarkResult()
            result.prefill_sweep.x_values.append(8)
            result.prefill_sweep.tps_values.append(16.0)
            result.prefill_sweep.time_values.append(0.5)
            result.prefill_sweep.avg_total_token_latency_values.append(0.0625)
            result.prefill_sweep.avg_npu_token_latency_values.append(None)
            result.decode_sweep.x_values.append(4)
            result.decode_sweep.tps_values.append(8.0)
            result.decode_sweep.time_values.append(0.5)
            result.decode_sweep.avg_total_token_latency_values.append(0.125)
            result.decode_sweep.avg_npu_token_latency_values.append(None)
            return result

    plot_path = tmp_path / "vlm_sweep.png"
    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: None)
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)

    args = argparse.Namespace(
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
        batch_size=1,
        warmup=0,
        repeat=1,
        prefill_range=(8, 8, 1),
        cache_lengths=[4],
        decode_window=2,
        image_resolutions=[224, 448],
        llm_resolution=224,
        prompt="Describe.",
        device_metrics=False,
        json=None,
        csv=None,
        plot=str(plot_path),
        device_backend="none",
    )

    assert tps_cli._run_vlm_sweep(args) == 0
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0


def test_run_vlm_sweep_ignores_tracker_stop_errors(monkeypatch):
    """Verify tracker cleanup failures do not abort completed VLM sweeps."""
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=1)))

    class _FailingStopTracker:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            raise RuntimeError("tracker stop failed")

        def get_metric(self) -> dict[str, float]:
            return {}

        def get_total_power_trace(self) -> list[tuple[float, float]]:
            return []

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure_vision(self, **kwargs) -> list[tuple[float, float]]:
            return [(float(kwargs["image_resolution"]) / 1000.0, 10.0)]

        def measure_llm_full(self, **kwargs) -> BenchmarkResult:
            del kwargs
            result = BenchmarkResult()
            result.prefill_sweep.x_values.append(8)
            result.prefill_sweep.tps_values.append(16.0)
            result.prefill_sweep.time_values.append(0.5)
            result.decode_sweep.x_values.append(4)
            result.decode_sweep.tps_values.append(8.0)
            result.decode_sweep.time_values.append(0.5)
            return result

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: _FailingStopTracker())
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)

    args = argparse.Namespace(
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
        batch_size=1,
        warmup=0,
        repeat=1,
        prefill_range=(8, 8, 1),
        cache_lengths=[4],
        decode_window=2,
        image_resolutions=[224],
        llm_resolution=224,
        prompt="Describe.",
        device_metrics=False,
        json=None,
        csv=None,
        plot=None,
        device_backend="npu",
    )

    assert tps_cli._run_vlm_sweep(args) == 0


def test_resolve_config_vocab_size_uses_top_level_config():
    config = _DummyConfig(vocab_size=128)

    assert _resolve_config_vocab_size(config) == 128


def test_resolve_config_vocab_size_uses_text_config():
    config = _DummyConfig(text_config=_DummyConfig(vocab_size=256))

    assert _resolve_config_vocab_size(config) == 256


def test_resolve_config_vocab_size_requires_vocab_size():
    with pytest.raises(AttributeError, match="vocab_size"):
        _resolve_config_vocab_size(_DummyConfig())


def test_supports_fake_decode_prefill_requires_npu_and_cache_model():
    assert _supports_fake_decode_prefill(_DummyNPUModel()) is True
    assert _supports_fake_decode_prefill(_DummyNonNPUModel()) is False


def test_supports_fake_decode_prefill_detects_nested_vlm_language_model():
    assert _supports_fake_decode_prefill(_DummyVLMContainer(_DummyNPUModel())) is True


def test_npu_timing_target_detects_nested_vlm_language_model():
    model = _DummyNestedTimingModel()

    assert _get_npu_timing_target(model) is model.model.language_model


def test_npu_timing_target_detects_aggregate_timing_api_without_backend_marker():
    """Detect EAGLE-3-style top-level aggregate timing APIs."""
    model = _DummyAggregateTimingModel()

    assert _get_npu_timing_target(model) is model


def test_tps_measure_full_uses_fake_decode_prefill_for_npu_models():
    measurer = _RoutingTPSMeasurer(_DummyNPUModel())

    result = measurer.measure_full(prefill_range=(1, 1, 1), cache_lengths=[4], decode_window=2)

    assert measurer.calls == [("real", 1, 1), ("fake", 4, 2)]
    assert result.decode_sweep.x_values == [4]
    assert result.decode_sweep.tps_values == [2.0]
    assert result.decode_prefill_modes == ["fake"]


def test_tps_measure_full_keeps_real_decode_prefill_for_non_npu_models():
    measurer = _RoutingTPSMeasurer(_DummyNonNPUModel())

    result = measurer.measure_full(prefill_range=(1, 1, 1), cache_lengths=[4], decode_window=2)

    assert measurer.calls == [("real", 1, 1), ("real", 4, 2)]
    assert result.decode_sweep.x_values == [4]
    assert result.decode_sweep.tps_values == [2.0]
    assert result.decode_prefill_modes == ["real"]


def test_benchmark_result_iter_rows_includes_decode_prefill_mode():
    result = BenchmarkResult()
    result.decode_sweep.x_values.append(4)
    result.decode_sweep.tps_values.append(2.0)
    result.decode_sweep.time_values.append(1.0)
    result.decode_sweep.avg_total_token_latency_values.append(0.5)
    result.decode_sweep.avg_npu_token_latency_values.append(0.25)
    result.decode_prefill_modes.append("fake")

    rows = list(BenchmarkResult.iter_rows("dummy", result))

    assert rows == [
        {
            "model": "dummy",
            "phase": "decode",
            "tokens": 4,
            "tps": 2.0,
            "time_ms": 1000.0,
            "avg_total_token_latency_ms": 500.0,
            "avg_npu_token_latency_ms": 250.0,
            "avg_npu_token_latency_pct": 50.0,
            "decode_prefill_mode": "fake",
        }
    ]


def test_npu_latency_pct_handles_supported_and_unsupported_values():
    assert npu_latency_pct(0.5, 0.25) == pytest.approx(50.0)
    assert npu_latency_pct(0.0, 0.25) is None
    assert npu_latency_pct(None, 0.25) is None
    assert npu_latency_pct(0.5, None) is None


def test_tps_measurer_reads_eagle3_acceptance_stats_from_public_property_only():
    """Read acceptance stats from `last_eagle3_acceptance_stats` public property."""
    pipeline = SimpleNamespace(model=_DummyEagle3AcceptanceModel(), tokenizer=_DummyTokenizer())
    measurer = TPSMeasurer(pipeline)

    stats = measurer._get_eagle3_acceptance_stats()

    assert stats == {
        "acceptance_steps": 3,
        "acceptance_tokens_sum": 7,
        "acceptance_tokens_avg": pytest.approx(7.0 / 3.0),
        "acceptance_ratio": 0.5,
    }


def test_single_measurement_json_exposes_npu_latency_pct():
    measurement = SingleMeasurement(
        num_prefill=4,
        num_decode=2,
        prefill_latency=2.0,
        prefill_tps=2.0,
        decode_duration=1.0,
        decode_tps=2.0,
        total_time=3.0,
        avg_total_prefill_token_latency=0.5,
        avg_npu_prefill_token_latency=0.25,
        avg_total_decode_token_latency=0.5,
        avg_npu_decode_token_latency=0.1,
        prefill_npu_latency_pct=npu_latency_pct(0.5, 0.25),
        decode_npu_latency_pct=npu_latency_pct(0.5, 0.1),
        total_npu_latency_pct=npu_latency_pct(3.0, 1.2),
    )

    assert measurement.prefill_npu_latency_pct == pytest.approx(50.0)
    assert measurement.decode_npu_latency_pct == pytest.approx(20.0)
    assert measurement.total_npu_latency_pct == pytest.approx(40.0)


def test_mobilint_model_mixin_aggregates_npu_timing_without_token_records():
    model = object.__new__(MobilintModelMixin)

    model.reset_npu_timing()
    model._record_npu_timing("prefill", 0.25)
    model._record_npu_timing("decode", 0.1)
    model._record_npu_timing("decode", 0.2)

    timing = model.get_npu_timing()

    assert timing["prefill_time"] == pytest.approx(0.25)
    assert timing["decode_time"] == pytest.approx(0.3)
    assert timing["prefill_calls"] == 1
    assert timing["decode_calls"] == 2
    assert set(timing) == {"prefill_time", "decode_time", "prefill_calls", "decode_calls"}


def test_single_measurement_populates_nanosecond_timings():
    measurement = SingleMeasurement(
        num_prefill=4,
        num_decode=2,
        prefill_latency=2.0,
        prefill_tps=2.0,
        decode_duration=1.0,
        decode_tps=2.0,
        total_time=3.0,
        avg_total_prefill_token_latency=0.5,
        avg_npu_prefill_token_latency=0.25,
        avg_total_decode_token_latency=0.5,
        avg_npu_decode_token_latency=0.1,
        npu_prefill_time=1.0,
        npu_decode_time=0.2,
    )

    assert measurement.prefill_latency_ns == 2_000_000_000
    assert measurement.decode_duration_ns == 1_000_000_000
    assert measurement.total_time_ns == 3_000_000_000
    assert measurement.avg_total_prefill_token_latency_ns == 500_000_000
    assert measurement.avg_npu_prefill_token_latency_ns == 250_000_000
    assert measurement.avg_total_decode_token_latency_ns == 500_000_000
    assert measurement.avg_npu_decode_token_latency_ns == 100_000_000
    assert measurement.npu_prefill_time_ns == 1_000_000_000
    assert measurement.npu_decode_time_ns == 200_000_000


def test_single_measurement_preserves_explicit_nanosecond_timings():
    measurement = SingleMeasurement(
        num_prefill=4,
        num_decode=2,
        prefill_latency=2.0,
        prefill_tps=2.0,
        decode_duration=1.0,
        decode_tps=2.0,
        total_time=3.0,
        avg_total_prefill_token_latency=0.5,
        avg_npu_prefill_token_latency=None,
        avg_total_decode_token_latency=0.5,
        avg_npu_decode_token_latency=None,
        prefill_latency_ns=123,
        decode_duration_ns=456,
        total_time_ns=579,
        avg_total_prefill_token_latency_ns=12,
        avg_total_decode_token_latency_ns=45,
    )

    assert measurement.prefill_latency_ns == 123
    assert measurement.decode_duration_ns == 456
    assert measurement.total_time_ns == 579
    assert measurement.avg_total_prefill_token_latency_ns == 12
    assert measurement.avg_total_decode_token_latency_ns == 45
    assert measurement.avg_npu_prefill_token_latency_ns is None
    assert measurement.avg_npu_decode_token_latency_ns is None


def test_cli_iter_rows_for_csv_includes_npu_latency_pct():
    result = BenchmarkResult()
    result.prefill_sweep.x_values.append(8)
    result.prefill_sweep.tps_values.append(16.0)
    result.prefill_sweep.time_values.append(0.8)
    result.prefill_sweep.avg_total_token_latency_values.append(0.1)
    result.prefill_sweep.avg_npu_token_latency_values.append(0.025)

    rows = list(tps_cli._iter_rows_for_csv(result))

    assert rows == [
        {
            "phase": "prefill",
            "tokens": 8,
            "tps": 16.0,
            "time_ms": 800.0,
            "avg_total_token_latency_ms": 100.0,
            "avg_npu_token_latency_ms": 25.0,
            "avg_npu_token_latency_pct": 25.0,
        }
    ]


def test_cli_write_csv_uses_union_fieldnames(tmp_path):
    csv_path = tmp_path / "vlm.csv"

    tps_cli._write_csv(
        str(csv_path),
        [
            {"type": "vision", "vision_encode_ms": 1.0},
            {"type": "llm", "llm_prefill_npu_latency_pct": 50.0},
        ],
    )

    content = csv_path.read_text(encoding="utf-8")
    assert "llm_prefill_npu_latency_pct" in content
    assert "50.0" in content


def test_vlm_fake_prefill_decode_counts_each_decode_token():
    language_model = _DummyVLMLanguageModel()
    measurer = _DummyVLMTPSMeasurer(language_model)

    result = measurer._measure_llm_decode_with_fake_prefill(cache_len=8, num_decode=4)

    assert language_model.cache.prefill_length == 8
    assert language_model.calls == 4
    assert result.num_decode == 4
    assert result.decode_tps > 0.0
    assert result.npu_decode_time == pytest.approx(1.0)
    assert result.avg_npu_decode_token_latency == pytest.approx(0.25)


def test_vlm_batched_llm_decode_count_subtracts_prompt_length():
    language_model = _DummyVLMLanguageModel()
    num_decode = 4
    measurer = _DummyBatchedVLMTPSMeasurer(language_model, generated_tokens=num_decode + 1)
    inputs_embeds = torch.zeros((2, 8, 4), dtype=torch.float32)

    result = measurer._measure_llm_once(inputs_embeds, num_decode=num_decode)

    assert result.num_prefill == 8
    assert result.num_decode == num_decode
    assert result.decode_tps > 0.0


def test_vlm_measure_vision_reports_per_image_latency_for_batch():
    language_model = _DummyVLMLanguageModel()
    measurer = _DummyVisionLatencyVLMTPSMeasurer(language_model, batch_latency=2.0)

    result = measurer.measure_vision(image_resolution=224, repeat=1, prompt="Describe it.", batch_size=4)

    assert result == [(pytest.approx(0.5), pytest.approx(2.0))]


def test_vlm_measure_reports_per_image_vision_latency_for_batch():
    language_model = _DummyVLMLanguageModel()
    measurer = _DummyVisionLatencyVLMTPSMeasurer(language_model, batch_latency=2.0)

    result = measurer.measure(image_resolution=224, num_decode=4, repeat=1, prompt="Describe it.", batch_size=4)

    assert len(result) == 1
    assert result[0].vision_encode_latency == pytest.approx(0.5)
    assert result[0].vision_fps == pytest.approx(2.0)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_text_generate_supports_phase_callbacks_for_all_batch_sizes(batch_size: int):
    model = _DummyGenerateNPUModel() if batch_size == 1 else _DummyBatchedGenerateNPUModel()
    measurer = TPSMeasurer(_DummyTextPipeline(model))
    callbacks: list[str] = []

    result = measurer.measure(
        num_prefill=8,
        num_decode=4,
        batch_size=batch_size,
        on_prefill_start=lambda: callbacks.append("prefill_start"),
        on_prefill_end=lambda: callbacks.append("prefill_end"),
        on_decode_start=lambda: callbacks.append("decode_start"),
        on_decode_end=lambda: callbacks.append("decode_end"),
    )

    assert callbacks == ["prefill_start", "prefill_end", "decode_start", "decode_end"]
    assert result.num_decode == 4


def test_run_text_measure_starts_phase_trackers_for_resolved_batch(monkeypatch, tmp_path):
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    events: list[str] = []
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=2)))

    class _FakeTracker:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self) -> None:
            events.append(f"{self.name}:start")

        def stop(self) -> None:
            events.append(f"{self.name}:stop")

        def get_metric(self) -> dict[str, float]:
            return {"avg_power_w": 2.0, "p99_power_w": 3.0}

        def get_total_power_trace(self) -> list[tuple[float, float]]:
            return [(0.0, 2.0), (1.0, 2.0)]

    class _FakeTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure(self, **kwargs) -> SingleMeasurement:
            if kwargs.get("on_prefill_start") is not None:
                kwargs["on_prefill_start"]()
                kwargs["on_prefill_end"]()
                kwargs["on_decode_start"]()
                kwargs["on_decode_end"]()
            return SingleMeasurement(
                num_prefill=kwargs["num_prefill"],
                num_decode=kwargs["num_decode"],
                prefill_latency=1.0,
                prefill_tps=2.0,
                decode_duration=1.0,
                decode_tps=2.0,
                total_time=2.0,
                avg_total_prefill_token_latency=1.0,
                avg_npu_prefill_token_latency=None,
                avg_total_decode_token_latency=1.0,
                avg_npu_decode_token_latency=None,
            )

    json_path = tmp_path / "measure.json"
    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(
        tps_cli,
        "_build_phase_trackers",
        lambda args, pipeline: (_FakeTracker("prefill"), _FakeTracker("decode")),
    )
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "TPSMeasurer", _FakeTPSMeasurer)

    args = argparse.Namespace(
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
        batch_size=None,
        warmup=0,
        repeat=1,
        prefill=8,
        decode=2,
        npu_prefill_chunk_size=None,
        trace=None,
        device_metrics=True,
        json=str(json_path),
        device_backend="npu",
    )

    assert tps_cli._run_text_measure(args) == 0
    assert events[:4] == ["prefill:start", "prefill:stop", "decode:start", "decode:stop"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["batch_size"] == 2
    assert payload["device_time_series_runs"][0]["prefill"]["power_w"] == [
        {"timestamp_s": 0.0, "value": 2.0},
        {"timestamp_s": 1.0, "value": 2.0},
    ]


def test_run_vlm_measure_forwards_npu_prefill_chunk_size(monkeypatch):
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    calls: list[dict[str, object]] = []
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=2)))

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure_vision(self, **kwargs) -> list[tuple[float, float]]:
            calls.append({"phase": "vision", **kwargs})
            return [(1.0, 2.0)]

        def measure_llm_full(self, **kwargs) -> BenchmarkResult:
            calls.append({"phase": "llm", **kwargs})
            result = BenchmarkResult()
            result.prefill_sweep.x_values.append(kwargs["prefill_range"][0])
            result.prefill_sweep.tps_values.append(8.0)
            result.prefill_sweep.time_values.append(1.0)
            result.prefill_sweep.avg_total_token_latency_values.append(0.125)
            result.prefill_sweep.avg_npu_token_latency_values.append(None)
            result.decode_sweep.x_values.append(kwargs["cache_lengths"][0])
            result.decode_sweep.tps_values.append(2.0)
            result.decode_sweep.time_values.append(1.0)
            result.decode_sweep.avg_total_token_latency_values.append(0.5)
            result.decode_sweep.avg_npu_token_latency_values.append(None)
            return result

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: None)
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)

    args = argparse.Namespace(
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
        batch_size=None,
        warmup=1,
        repeat=1,
        image_resolution=224,
        prefill=8,
        decode=2,
        prompt="Describe the image.",
        npu_prefill_chunk_size=64,
        device_metrics=False,
        json=None,
        device_backend="none",
    )

    assert tps_cli._run_vlm_measure(args) == 0
    llm_calls = [call for call in calls if call["phase"] == "llm"]
    assert [call["npu_prefill_chunk_size"] for call in llm_calls] == [64, 64]
    assert [call["batch_size"] for call in llm_calls] == [2, 2]
    assert [call["prefill_range"] for call in llm_calls] == [(8, 8, 8), (8, 8, 8)]
    assert [call["cache_lengths"] for call in llm_calls] == [[8], [8]]


def test_run_vlm_measure_scales_total_ms_by_batch_size(monkeypatch, tmp_path):
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=4)))

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure_vision(self, **kwargs) -> list[tuple[float, float]]:
            return [(1.0, 4.0)]

        def measure_llm_full(self, **kwargs) -> BenchmarkResult:
            result = BenchmarkResult()
            result.prefill_sweep.x_values.append(kwargs["prefill_range"][0])
            result.prefill_sweep.tps_values.append(8.0)
            result.prefill_sweep.time_values.append(1.0)
            result.prefill_sweep.avg_total_token_latency_values.append(0.125)
            result.prefill_sweep.avg_npu_token_latency_values.append(None)
            result.decode_sweep.x_values.append(kwargs["cache_lengths"][0])
            result.decode_sweep.tps_values.append(2.0)
            result.decode_sweep.time_values.append(1.0)
            result.decode_sweep.avg_total_token_latency_values.append(0.5)
            result.decode_sweep.avg_npu_token_latency_values.append(None)
            return result

    json_path = tmp_path / "vlm_measure.json"
    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: None)
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)

    args = argparse.Namespace(
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
        batch_size=4,
        warmup=0,
        repeat=1,
        image_resolution=224,
        prefill=8,
        decode=2,
        prompt="Describe the image.",
        npu_prefill_chunk_size=None,
        device_metrics=False,
        json=str(json_path),
        device_backend="none",
    )

    assert tps_cli._run_vlm_measure(args) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["vision_encode_ms"]["mean"] == pytest.approx(1000.0)
    assert payload["summary"]["total_ms"]["mean"] == pytest.approx(6000.0)


def test_run_vlm_measure_sums_phase_trace_energy_for_total_energy(monkeypatch, tmp_path):
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=4)))

    class _FakeTracker:
        def __init__(self, power_trace: list[tuple[float, float]]) -> None:
            self._power_trace = power_trace

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def get_metric(self) -> dict[str, float]:
            return {"avg_power_w": 2.0}

        def get_total_power_trace(self) -> list[tuple[float, float]]:
            return self._power_trace

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure_vision(self, **kwargs) -> list[tuple[float, float]]:
            return [(1.0, 4.0)]

        def measure_llm_full(self, **kwargs) -> BenchmarkResult:
            result = BenchmarkResult()
            result.prefill_sweep.x_values.append(kwargs["prefill_range"][0])
            result.prefill_sweep.tps_values.append(8.0)
            result.prefill_sweep.time_values.append(1.0)
            result.prefill_sweep.avg_total_token_latency_values.append(0.125)
            result.prefill_sweep.avg_npu_token_latency_values.append(None)
            result.decode_sweep.x_values.append(kwargs["cache_lengths"][0])
            result.decode_sweep.tps_values.append(2.0)
            result.decode_sweep.time_values.append(1.0)
            result.decode_sweep.avg_total_token_latency_values.append(0.5)
            result.decode_sweep.avg_npu_token_latency_values.append(None)
            return result

    json_path = tmp_path / "vlm_measure_energy.json"
    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: _FakeTracker([(0.0, 2.0), (1.0, 2.0)]))
    monkeypatch.setattr(
        tps_cli,
        "_build_phase_trackers",
        lambda args, pipeline: (_FakeTracker([(0.0, 2.0), (2.0, 2.0)]), _FakeTracker([(0.0, 2.0), (3.0, 2.0)])),
    )
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)

    args = argparse.Namespace(
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
        batch_size=4,
        warmup=0,
        repeat=1,
        image_resolution=224,
        prefill=8,
        decode=2,
        prompt="Describe the image.",
        npu_prefill_chunk_size=None,
        device_metrics=True,
        json=str(json_path),
        device_backend="npu",
    )

    assert tps_cli._run_vlm_measure(args) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["device_runs"][0]["vision_energy_j"] == pytest.approx(2.0)
    assert payload["device_runs"][0]["llm_prefill_energy_j"] == pytest.approx(4.0)
    assert payload["device_runs"][0]["llm_decode_energy_j"] == pytest.approx(6.0)
    assert payload["device_runs"][0]["llm_total_energy_j"] == pytest.approx(10.0)
    assert payload["device_runs"][0]["total_energy_j"] == pytest.approx(12.0)
    assert payload["summary"]["vision_energy_j"]["mean"] == pytest.approx(2.0)
    assert payload["summary"]["llm_prefill_energy_j"]["mean"] == pytest.approx(4.0)
    assert payload["summary"]["llm_decode_energy_j"]["mean"] == pytest.approx(6.0)
    assert payload["summary"]["llm_total_energy_j"]["mean"] == pytest.approx(10.0)
    assert payload["summary"]["total_energy_j"]["mean"] == pytest.approx(12.0)


def test_run_vlm_measure_ignores_tracker_stop_errors(monkeypatch):
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=1)))

    class _FailingStopTracker:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            raise RuntimeError("tracker stop failed")

        def get_metric(self) -> dict[str, float]:
            return {}

        def get_total_power_trace(self) -> list[tuple[float, float]]:
            return []

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure_vision(self, **kwargs) -> list[tuple[float, float]]:
            return [(1.0, 1.0)]

        def measure_llm_full(self, **kwargs) -> BenchmarkResult:
            result = BenchmarkResult()
            result.prefill_sweep.x_values.append(kwargs["prefill_range"][0])
            result.prefill_sweep.tps_values.append(8.0)
            result.prefill_sweep.time_values.append(1.0)
            result.prefill_sweep.avg_total_token_latency_values.append(0.125)
            result.prefill_sweep.avg_npu_token_latency_values.append(None)
            result.decode_sweep.x_values.append(kwargs["cache_lengths"][0])
            result.decode_sweep.tps_values.append(2.0)
            result.decode_sweep.time_values.append(1.0)
            result.decode_sweep.avg_total_token_latency_values.append(0.5)
            result.decode_sweep.avg_npu_token_latency_values.append(None)
            return result

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: _FailingStopTracker())
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(benchmark_utils, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)

    args = argparse.Namespace(
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
        batch_size=1,
        warmup=0,
        repeat=1,
        image_resolution=224,
        prefill=8,
        decode=2,
        prompt="Describe the image.",
        npu_prefill_chunk_size=None,
        device_metrics=False,
        json=None,
        device_backend="npu",
    )

    assert tps_cli._run_vlm_measure(args) == 0


def test_text_fake_prefill_generate_uses_cache_length_plus_decode_seed():
    model = _DummyGenerateNPUModel()
    measurer = TPSMeasurer(_DummyTextPipeline(model))

    result = measurer.measure_decode_with_fake_prefill(cache_len=8, num_decode=4)

    assert model.generate_kwargs is not None
    assert model.generate_kwargs["input_ids"].shape == (1, 9)
    assert model.generate_kwargs["past_key_values"].get_seq_length() == 8
    assert model.generate_kwargs["min_new_tokens"] == 4
    assert model.generate_kwargs["max_new_tokens"] == 4
    assert result.decode_prefill_mode == "fake"


def test_temporarily_sanitize_generation_config_restores_sampling_flags():
    model = _DummyGenerateNPUModel()

    with _temporarily_sanitize_generation_config(model):
        assert model.generation_config.do_sample is False
        assert model.generation_config.temperature is None
        assert model.generation_config.top_p is None
        assert model.generation_config.top_k is None

    assert model.generation_config.temperature == 0.6
    assert model.generation_config.top_p == 0.95
    assert model.generation_config.top_k == 20


def test_text_fake_prefill_generate_sanitizes_model_config_without_kwargs():
    model = _DummyGenerateNPUModel()
    measurer = TPSMeasurer(_DummyTextPipeline(model))

    measurer.measure_decode_with_fake_prefill(cache_len=8, num_decode=4)

    assert "generation_config" not in model.generate_kwargs
    assert "temperature" not in model.generate_kwargs
    assert "top_p" not in model.generate_kwargs
    assert "top_k" not in model.generate_kwargs
    assert model.generation_config_during_generate == {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }
    assert model.generation_config.temperature == 0.6
    assert model.generation_config.top_p == 0.95
    assert model.generation_config.top_k == 20


def test_resolve_image_features_tensor_uses_tensor():
    image_features = torch.zeros(2, 4)

    assert _resolve_image_features_tensor(image_features) is image_features


def test_resolve_image_features_tensor_uses_pooler_output():
    pooler_output = torch.zeros(2, 4)
    image_features = _DummyVisionOutput(pooler_output=pooler_output)

    assert _resolve_image_features_tensor(image_features) is pooler_output


def test_resolve_image_features_tensor_concatenates_pooler_output_tuple():
    first = torch.zeros(2, 4)
    second = torch.ones(3, 4)
    image_features = _DummyVisionOutput(pooler_output=(first, second))

    resolved = _resolve_image_features_tensor(image_features)

    assert torch.equal(resolved, torch.cat((first, second), dim=0))


def test_resolve_image_features_tensor_uses_tuple_tensor():
    image_features = torch.zeros(2, 4)

    assert _resolve_image_features_tensor((image_features, None)) is image_features


def test_resolve_image_features_tensor_uses_nested_qwen3_vl_tuple():
    image_features = torch.zeros(2, 4)
    deepstack_features = [torch.ones(2, 4), torch.full((2, 4), 2.0), torch.full((2, 4), 3.0)]

    assert _resolve_image_features_tensor(((image_features,), deepstack_features)) is image_features


def test_resolve_image_features_tensor_requires_tensor():
    with pytest.raises(TypeError, match="image feature tensor"):
        _resolve_image_features_tensor(_DummyVisionOutput())


@pytest.mark.parametrize(
    ("model_cls", "method_name"),
    [
        (MobilintQwen2VLForConditionalGeneration, "forward"),
        (MobilintQwen2VLForConditionalGeneration, "prepare_inputs_for_generation"),
    ],
)
def test_mobilint_generation_hooks_accept_count_npu_time(model_cls, method_name: str):
    signature = inspect.signature(getattr(model_cls, method_name))

    assert "count_npu_time" in signature.parameters


@pytest.mark.parametrize(
    "model_path",
    [
        "mblt_model_zoo.hf_transformers.models.cohere2.modeling_cohere2:MobilintCohere2ForCausalLM",
        "mblt_model_zoo.hf_transformers.models.exaone.modeling_exaone:MobilintExaoneForCausalLM",
        "mblt_model_zoo.hf_transformers.models.exaone4.modeling_exaone4:MobilintExaone4ForCausalLM",
        "mblt_model_zoo.hf_transformers.models.llama.modeling_llama:MobilintLlamaForCausalLM",
        "mblt_model_zoo.hf_transformers.models.qwen2.modeling_qwen2:MobilintQwen2ForCausalLM",
        "mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl:MobilintQwen2VLForConditionalGeneration",
        "mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl:MobilintQwen2VLTextModel",
        "mblt_model_zoo.hf_transformers.models.qwen3.modeling_qwen3:MobilintQwen3ForCausalLM",
    ],
)
def test_mobilint_generation_hooks_expose_inputs_embeds(model_path: str):
    module_name, class_name = model_path.split(":")
    model_cls = getattr(importlib.import_module(module_name), class_name)

    forward_signature = inspect.signature(model_cls.forward)
    prepare_signature = inspect.signature(model_cls.prepare_inputs_for_generation)

    assert "inputs_embeds" in forward_signature.parameters
    assert "inputs_embeds" in prepare_signature.parameters


def test_blip_text_generation_hook_exposes_inputs_embeds():
    signature = inspect.signature(MobilintBlipTextLMHeadModel.prepare_inputs_for_generation)

    assert "inputs_embeds" in signature.parameters


def test_mobilint_generation_mixin_preserves_benchmark_kwargs(monkeypatch: pytest.MonkeyPatch):
    class _DummyGenerationModel(MobilintGenerationMixin):
        pass

    def _base_prepare_inputs_for_generation(*args, **kwargs):
        del args, kwargs
        return {"input_ids": torch.tensor([[1]])}

    monkeypatch.setattr(
        GenerationMixin,
        "prepare_inputs_for_generation",
        _base_prepare_inputs_for_generation,
    )

    model_inputs = _DummyGenerationModel().prepare_inputs_for_generation(
        torch.tensor([[1]]),
        count_npu_time=True,
        npu_prefill_chunk_size=64,
    )

    assert model_inputs["count_npu_time"] is True
    assert model_inputs["npu_prefill_chunk_size"] == 64


def test_text_model_prepare_inputs_preserves_npu_prefill_chunk_size(monkeypatch: pytest.MonkeyPatch):
    """Verify text-only model MRO preserves TPS prefill chunk kwargs for generation."""
    from mblt_model_zoo.hf_transformers.models.llama.modeling_llama import MobilintLlamaForCausalLM

    signature = inspect.signature(MobilintLlamaForCausalLM.prepare_inputs_for_generation)

    assert "count_npu_time" in signature.parameters
    assert "npu_prefill_chunk_size" in signature.parameters

    def _base_prepare_inputs_for_generation(*args, **kwargs):
        del args, kwargs
        return {"input_ids": torch.tensor([[1]])}

    monkeypatch.setattr(
        GenerationMixin,
        "prepare_inputs_for_generation",
        _base_prepare_inputs_for_generation,
    )
    model = object.__new__(MobilintLlamaForCausalLM)

    model_inputs = model.prepare_inputs_for_generation(
        torch.tensor([[1]]),
        count_npu_time=True,
        npu_prefill_chunk_size=64,
    )

    assert model_inputs["count_npu_time"] is True
    assert model_inputs["npu_prefill_chunk_size"] == 64


def test_qwen2_vl_prepare_inputs_preserves_npu_prefill_chunk_size(monkeypatch: pytest.MonkeyPatch):
    signature = inspect.signature(MobilintQwen2VLForConditionalGeneration.prepare_inputs_for_generation)

    assert "npu_prefill_chunk_size" in signature.parameters

    def _base_prepare_inputs_for_generation(*args, **kwargs):
        del args, kwargs
        return {"input_ids": torch.tensor([[1]])}

    monkeypatch.setattr(
        "mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl."
        "Qwen2VLForConditionalGeneration.prepare_inputs_for_generation",
        _base_prepare_inputs_for_generation,
    )
    model = object.__new__(MobilintQwen2VLForConditionalGeneration)

    model_inputs = model.prepare_inputs_for_generation(
        torch.tensor([[1]]),
        count_npu_time=True,
        npu_prefill_chunk_size=64,
    )

    assert model_inputs["count_npu_time"] is True
    assert model_inputs["npu_prefill_chunk_size"] == 64


@pytest.mark.parametrize("spec", ["", "1", "1:2", "1:2:0", "2:1:1", "a:b:c"])
def test_cli_tps_invalid_range_exits(spec: str):
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "tps",
                "sweep",
                "--model",
                "mobilint/Llama-3.2-1B-Instruct",
                "--prefill-range",
                spec,
                "--no-plot",
            ]
        )
    assert excinfo.value.code == 2


def test_cli_tps_device_backend_none_parsing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--device-backend",
            "none",
        ]
    )
    assert args.device_backend == "none"


def test_cli_tps_vlm_build_pipeline_uses_prefixed_core_kwargs(monkeypatch: pytest.MonkeyPatch):
    """Verify VLM TPS pipeline maps core-mode kwargs to vision/text-prefixed config fields."""
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("transformers.pipeline", _fake_pipeline)

    tps_cli._build_pipeline(
        task="image-text-to-text",
        model="mobilint/vlm-a",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode="global8",
        target_cores=None,
        target_clusters=None,
    )

    model_kwargs = captured["model_kwargs"]
    assert model_kwargs == {
        "vision_core_mode": "global8",
        "vision_target_clusters": [0, 1],
        "text_core_mode": "global8",
        "text_target_clusters": [0, 1],
    }
    assert "core_mode" not in model_kwargs
    assert "target_clusters" not in model_kwargs


def test_cli_tps_vlm_core_kwargs_preserve_explicit_targets():
    """Verify explicit TPS target core-mode values are prefixed for VLM tasks."""
    kwargs = tps_cli._apply_vlm_core_mode_model_kwargs(
        {},
        "single",
        target_cores=["0:2"],
        target_clusters=[1],
    )

    assert kwargs == {
        "vision_core_mode": "single",
        "vision_target_cores": ["0:2"],
        "vision_target_clusters": [1],
        "text_core_mode": "single",
        "text_target_cores": ["0:2"],
        "text_target_clusters": [1],
    }


def test_cli_tps_eagle3_build_pipeline_uses_prefixed_backend_kwargs(monkeypatch: pytest.MonkeyPatch):
    """Verify EAGLE-3 TPS pipeline maps prefixed MXQ and embedding kwargs directly."""
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("transformers.pipeline", _fake_pipeline)

    tps_cli._build_pipeline(
        task="text-generation",
        model="mobilint/EAGLE3-JPharmatron-7B",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(
            base_embedding_path="base.pt",
            draft_embedding_path="draft.pt",
            base_mxq_path="base.mxq",
            draft_mxq_path="draft.mxq",
            fc_mxq_path="fc.mxq",
        ),
        mxq_path=None,
        core_mode="global4",
        target_cores=None,
        target_clusters=None,
    )

    model_kwargs = captured["model_kwargs"]
    assert model_kwargs["base_embedding_weight"] == "base.pt"
    assert model_kwargs["draft_embedding_weight"] == "draft.pt"
    assert model_kwargs["base_mxq_path"] == "base.mxq"
    assert model_kwargs["draft_mxq_path"] == "draft.mxq"
    assert model_kwargs["fc_mxq_path"] == "fc.mxq"
    assert model_kwargs["base_core_mode"] == "global4"
    assert model_kwargs["draft_core_mode"] == "global4"
    assert model_kwargs["fc_core_mode"] == "global4"


def test_cli_tps_eagle3_build_pipeline_preserves_empty_prefixed_targets(monkeypatch: pytest.MonkeyPatch):
    """Keep explicit empty prefixed targets instead of falling back to shared values."""
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("transformers.pipeline", _fake_pipeline)

    with pytest.warns(UserWarning) as warning_records:
        tps_cli._build_pipeline(
            task="text-generation",
            model="mobilint/EAGLE3-JPharmatron-7B",
            tokenizer=None,
            device="cpu",
            trust_remote_code=True,
            dtype=None,
            device_map=None,
            revision=None,
            embedding_weight=None,
            eagle3_options=tps_cli.Eagle3PipelineOptions(
                base_target_cores=[],
                base_target_clusters=[],
            ),
            mxq_path=None,
            core_mode="global8",
            target_cores=["0:0"],
            target_clusters=[0, 1],
        )

    warning_messages = [str(record.message) for record in warning_records]
    assert any("`--target-cores` and `--base-target-cores`" in message for message in warning_messages)
    assert any("`--target-clusters` and `--base-target-clusters`" in message for message in warning_messages)

    model_kwargs = captured["model_kwargs"]
    assert model_kwargs["base_core_mode"] == "global8"
    assert model_kwargs["base_target_cores"] == []
    assert model_kwargs["base_target_clusters"] == []


def test_cli_tps_eagle3_build_pipeline_prefers_prefixed_targets_over_shared(monkeypatch: pytest.MonkeyPatch):
    """Use prefixed EAGLE-3 target values when both shared and prefixed options are set."""
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("transformers.pipeline", _fake_pipeline)

    with pytest.warns(UserWarning) as warning_records:
        tps_cli._build_pipeline(
            task="text-generation",
            model="mobilint/EAGLE3-JPharmatron-7B",
            tokenizer=None,
            device="cpu",
            trust_remote_code=True,
            dtype=None,
            device_map=None,
            revision=None,
            embedding_weight=None,
            eagle3_options=tps_cli.Eagle3PipelineOptions(
                base_target_cores=["0:2"],
                draft_target_clusters=[1],
                fc_core_mode="single",
            ),
            mxq_path=None,
            core_mode="global8",
            target_cores=["0:0"],
            target_clusters=[0, 1],
        )

    warning_messages = [str(record.message) for record in warning_records]
    assert any("`--core-mode` and `--fc-core-mode`" in message for message in warning_messages)
    assert any("`--target-cores` and `--base-target-cores`" in message for message in warning_messages)
    assert any("`--target-clusters` and `--draft-target-clusters`" in message for message in warning_messages)

    model_kwargs = captured["model_kwargs"]
    assert model_kwargs["base_target_cores"] == ["0:2"]
    assert model_kwargs["draft_target_clusters"] == [1]
    assert model_kwargs["fc_core_mode"] == "single"


def test_benchmark_result_iter_rows_pads_missing_latency_values():
    """Verify combined rows survive old JSON-style sweeps without latency arrays."""
    result = BenchmarkResult()
    result.prefill_sweep.x_values.append(8)
    result.prefill_sweep.tps_values.append(10.0)
    result.prefill_sweep.time_values.append(0.8)
    result.decode_sweep.x_values.append(4)
    result.decode_sweep.tps_values.append(20.0)
    result.decode_sweep.time_values.append(0.2)

    rows = list(BenchmarkResult.iter_rows("dummy", result))

    assert len(rows) == 2
    assert rows[0]["phase"] == "prefill"
    assert rows[0]["avg_total_token_latency_ms"] is None
    assert rows[1]["phase"] == "decode"
    assert rows[1]["avg_npu_token_latency_pct"] is None


def test_cli_iter_rows_for_csv_pads_missing_latency_values():
    """Verify TPS CSV rows are emitted even when latency arrays are absent."""
    result = BenchmarkResult()
    result.prefill_sweep.x_values.append(8)
    result.prefill_sweep.tps_values.append(10.0)
    result.prefill_sweep.time_values.append(0.8)

    rows = list(tps_cli._iter_rows_for_csv(result))

    assert rows == [
        {
            "phase": "prefill",
            "tokens": 8,
            "tps": 10.0,
            "time_ms": 800.0,
            "avg_total_token_latency_ms": None,
            "avg_npu_token_latency_ms": None,
            "avg_npu_token_latency_pct": None,
        }
    ]


def test_cli_aggregate_sweep_results_tolerates_missing_latency_values():
    """Verify repeated TPS sweeps aggregate old results without latency arrays."""
    first = BenchmarkResult(prefill_sweep=SweepData(x_values=[8], tps_values=[10.0], time_values=[0.8]))
    second = BenchmarkResult(prefill_sweep=SweepData(x_values=[8], tps_values=[20.0], time_values=[1.2]))

    result = tps_cli._aggregate_sweep_results([first, second])

    assert result.prefill_sweep.tps_values == [15.0]
    assert result.prefill_sweep.avg_total_token_latency_values == [None]
    assert result.prefill_sweep.avg_npu_token_latency_values == [None]


def test_cli_attach_tps_per_w_uses_whole_sweep_token_scope():
    """Verify sweep TPS/W uses the same whole-phase scope as trace energy."""
    result = BenchmarkResult(
        prefill_sweep=SweepData(x_values=[8, 16], tps_values=[10.0, 20.0], time_values=[0.8, 0.8]),
        decode_sweep=SweepData(x_values=[32, 64], tps_values=[30.0, 40.0], time_values=[0.2, 0.2]),
    )

    tps_cli._attach_tps_per_w(
        result,
        prefill_energy=12.0,
        decode_energy=8.0,
        total_energy=20.0,
        batch_size=2,
        decode_window=4,
    )

    assert result.prefill_tps_per_w == pytest.approx(((8 + 16) * 2) / 12.0)
    assert result.decode_tps_per_w == pytest.approx((4 * 2 * 2) / 8.0)
    assert result.total_tps_per_w == pytest.approx((((8 + 16) * 2) + (4 * 2 * 2)) / 20.0)


def test_extract_device_metric_normalizes_tracker_023_shape():
    class _FakeTracker:
        def get_metric(self):
            return {
                "avg_power_w": 12,
                "p99_power_w": 15.5,
                "avg_utilization_pct": 80,
                "p99_utilization_pct": 95,
                "avg_temperature_c": "hot",
                "p99_temperature_c": 70,
                "avg_memory_used_mb": 1024,
                "p99_memory_used_mb": 2048,
                "total_memory_mb": 4096,
                "avg_memory_used_pct": 25,
                "p99_memory_used_pct": 50,
                "samples": 3,
                "gpu": {0: {"avg_power_w": 12.0}},
                "npu": {0: {"avg_power_w": 12.0}},
            }

    metric = extract_device_metric(_FakeTracker())

    assert tuple(metric.keys()) == DEVICE_METRIC_KEYS
    assert metric["avg_power_w"] == 12.0
    assert metric["avg_temperature_c"] is None
    assert all(value is None or isinstance(value, float) for value in metric.values())


def test_extract_device_time_series_uses_tracker_100_memory_trace_methods():
    class _FakeTracker:
        def get_total_power_trace(self):
            return [(1, 10.5), (2.0, 11)]

        def get_total_utilization_trace(self):
            return [(1.0, 70.0)]

        def get_temperature_trace(self):
            return [(1.0, 55.0), [2.0, 56.0]]

        def get_memory_used_trace(self):
            return [(1.0, 512), ("bad", 768), (2.0, "bad")]

        def get_memory_used_pct_trace(self):
            return [(1.0, 25.0)]

    series = extract_device_time_series(_FakeTracker())

    assert series["power_w"] == [
        {"timestamp_s": 1.0, "value": 10.5},
        {"timestamp_s": 2.0, "value": 11.0},
    ]
    assert series["utilization_pct"] == [{"timestamp_s": 1.0, "value": 70.0}]
    assert series["temperature_c"] == [{"timestamp_s": 1.0, "value": 55.0}]
    assert series["memory_used_mb"] == [{"timestamp_s": 1.0, "value": 512.0}]
    assert series["memory_used_pct"] == [{"timestamp_s": 1.0, "value": 25.0}]
    assert series["npu_power_w"] == []
    assert series["goldfinger_power_w"] == []


def test_parse_npu_rail_metrics_accepts_default_and_all():
    assert parse_npu_rail_metrics(None) == "npu"
    assert parse_npu_rail_metrics("") == "npu"
    assert parse_npu_rail_metrics(" all ") == "all"


def test_parse_npu_rail_metrics_accepts_comma_separated_rails():
    assert parse_npu_rail_metrics("npu,ddr,NPU") == ["npu", "ddr"]


def test_parse_npu_rail_metrics_rejects_unknown_rail():
    with pytest.raises(argparse.ArgumentTypeError, match="unknown NPU rail metric"):
        parse_npu_rail_metrics("cpu")


def test_build_device_tracker_forwards_npu_rail_metrics(monkeypatch: pytest.MonkeyPatch):
    created: dict[str, object] = {}

    class _FakeNPUDeviceTracker:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def get_metric(self) -> dict[str, float]:
            return {}

    fake_module = types.SimpleNamespace(NPUDeviceTracker=_FakeNPUDeviceTracker)
    monkeypatch.setitem(sys.modules, "mblt_tracker", fake_module)
    args = argparse.Namespace(
        device_metrics=True,
        device_backend="npu",
        device_npu_id=[0],
        device_npu_rail_metrics="all",
        device="cpu",
        device_gpu_id=None,
    )
    pipeline = SimpleNamespace(model=SimpleNamespace(npu_backend=object()))

    tracker = build_device_tracker(args, pipeline)

    assert isinstance(tracker, _FakeNPUDeviceTracker)
    assert created == {"interval": 1.0, "npu_id": 0, "rail_metrics": "all"}


def test_build_device_tracker_uses_gpu_interval_and_device_id(monkeypatch):
    """Verify GPU tracking uses the common interval and parses cuda device ids."""
    created = {}

    class _FakeGPUDeviceTracker:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def get_metric(self) -> dict[str, float]:
            return {}

    fake_module = types.SimpleNamespace(GPUDeviceTracker=_FakeGPUDeviceTracker)
    monkeypatch.setitem(sys.modules, "mblt_tracker", fake_module)
    args = argparse.Namespace(
        device_metrics=True,
        device_backend="gpu",
        device_npu_id=None,
        device_npu_rail_metrics="npu",
        device="cuda:1",
        device_gpu_id=None,
    )
    pipeline = SimpleNamespace(model=SimpleNamespace())

    tracker = build_device_tracker(args, pipeline)

    assert isinstance(tracker, _FakeGPUDeviceTracker)
    assert created == {"interval": 1.0, "gpu_id": 1}


def test_extract_device_time_series_uses_tracker_100_trace_methods():
    class _FakeTracker:
        def get_total_power_trace(self):
            return [(1.0, 100.0)]

        def get_total_utilization_trace(self):
            return [(2.0, 80.0)]

        def get_temperature_trace(self):
            return [(3.0, 50.0)]

        def get_npu_rail_power_trace(self):
            return [(4.0, 10.0)]

        def get_npu_power_trace(self):
            raise AssertionError("legacy NPU rail alias must not be used")

        def get_ddr_rail_power_trace(self):
            return [(5.0, 2.0)]

        def get_ddr_power_trace(self):
            raise AssertionError("legacy DDR rail alias must not be used")

        def get_pmic_rail_power_trace(self):
            return [(6.0, 3.0)]

        def get_pmic_power_trace(self):
            raise AssertionError("legacy PMIC rail alias must not be used")

        def get_goldfinger_rail_power_trace(self):
            return [(7.0, 4.0)]

        def get_goldfinger_power_trace(self):
            raise AssertionError("legacy goldfinger rail alias must not be used")

    series = extract_device_time_series(_FakeTracker())

    assert series["power_w"] == [{"timestamp_s": 1.0, "value": 100.0}]
    assert series["utilization_pct"] == [{"timestamp_s": 2.0, "value": 80.0}]
    assert series["temperature_c"] == [{"timestamp_s": 3.0, "value": 50.0}]
    assert series["npu_power_w"] == [{"timestamp_s": 4.0, "value": 10.0}]
    assert series["ddr_power_w"] == [{"timestamp_s": 5.0, "value": 2.0}]
    assert series["pmic_power_w"] == [{"timestamp_s": 6.0, "value": 3.0}]
    assert series["goldfinger_power_w"] == [{"timestamp_s": 7.0, "value": 4.0}]


def test_cli_tps_device_npu_id_parsing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--device-npu-id",
            "0,1",
        ]
    )

    assert args.device_npu_id == [0, 1]


def test_cli_tps_device_npu_rail_metrics_parsing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--device-npu-rail-metrics",
            "npu,ddr",
        ]
    )

    assert args.device_npu_rail_metrics == ["npu", "ddr"]


def test_cli_tps_device_npu_rail_metrics_default():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
        ]
    )

    assert args.device_npu_rail_metrics == "npu"


def test_cli_tps_device_npu_id_rejects_negative_values():
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "tps",
                "measure",
                "--model",
                "mobilint/Llama-3.2-1B-Instruct",
                "--device-npu-id",
                "-1",
            ]
        )

    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--prefill",
            "0",
        ],
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--decode",
            "-1",
        ],
        [
            "tps",
            "sweep",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--decode-window",
            "0",
            "--no-plot",
        ],
    ],
)
def test_cli_tps_positive_int_enforced(argv: list[str]):
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 2


def _trace_scope_args(**overrides):
    """Build a minimal TPS CLI namespace for trace-scope tests."""
    defaults = dict(
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
        warmup=1,
        repeat=2,
        prefill=8,
        decode=2,
        prefill_range=(8, 8, 1),
        cache_lengths=[4],
        decode_window=2,
        npu_prefill_chunk_size=None,
        trace="trace.json",
        device_metrics=False,
        json=None,
        csv=None,
        plot=None,
        device_backend="none",
        input_mode="random",
        prompt_text=None,
        prompt_file=None,
        prompt_file_strategy="first",
        prompt_file_seed=0,
        image_resolution=224,
        image_resolutions=[224],
        llm_resolution=224,
        prompt="Describe.",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _trace_single(num_prefill: int = 8, num_decode: int = 2) -> SingleMeasurement:
    """Return a deterministic single measurement."""
    return SingleMeasurement(
        num_prefill=num_prefill,
        num_decode=num_decode,
        prefill_latency=1.0,
        prefill_tps=float(num_prefill),
        decode_duration=1.0,
        decode_tps=float(num_decode),
        total_time=2.0,
        avg_total_prefill_token_latency=1.0 / num_prefill,
        avg_npu_prefill_token_latency=None,
        avg_total_decode_token_latency=1.0 / num_decode,
        avg_npu_decode_token_latency=None,
    )


def _trace_result(prefill: int = 8, decode: int = 4) -> BenchmarkResult:
    """Return a deterministic benchmark result."""
    result = BenchmarkResult()
    result.prefill_sweep.x_values.append(prefill)
    result.prefill_sweep.tps_values.append(float(prefill))
    result.prefill_sweep.time_values.append(1.0)
    result.prefill_sweep.avg_total_token_latency_values.append(1.0 / prefill)
    result.prefill_sweep.avg_npu_token_latency_values.append(None)
    result.decode_sweep.x_values.append(decode)
    result.decode_sweep.tps_values.append(float(decode))
    result.decode_sweep.time_values.append(1.0)
    result.decode_sweep.avg_total_token_latency_values.append(1.0 / decode)
    result.decode_sweep.avg_npu_token_latency_values.append(None)
    return result


def test_phase_trace_path_preserves_cli_trace_shape() -> None:
    """Phase trace paths should preserve the user-facing single --trace argument shape."""
    assert tps_cli._phase_trace_path("trace.json", "vision") == "trace.vision.json"
    assert tps_cli._phase_trace_path("outputs/trace.json", "llm").replace("\\", "/") == "outputs/trace.llm.json"
    assert tps_cli._phase_trace_path("trace", "vision") == "trace.vision"
    assert tps_cli._phase_trace_path(None, "vision") is None


def test_qbruntime_trace_ignores_nested_start_to_preserve_outer_file(monkeypatch):
    """Nested qbruntime trace requests should not overwrite the active outer trace file."""
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    calls: list[tuple[str, str | None]] = []
    fake_qbruntime = SimpleNamespace(
        start_tracing_events=lambda path: calls.append(("start", path)),
        stop_tracing_events=lambda: calls.append(("stop", None)),
    )

    monkeypatch.setitem(sys.modules, "qbruntime", fake_qbruntime)
    monkeypatch.setattr(benchmark_utils, "_ACTIVE_QBRUNTIME_TRACE_HANDLE", None)

    outer_handle = benchmark_utils.start_qbruntime_trace("outer.json")
    nested_handle = benchmark_utils.start_qbruntime_trace("nested.json")
    benchmark_utils.stop_qbruntime_trace(nested_handle)
    benchmark_utils.stop_qbruntime_trace(outer_handle)

    assert calls == [("start", "outer.json"), ("stop", None)]
    assert benchmark_utils._ACTIVE_QBRUNTIME_TRACE_HANDLE is None


def test_run_text_measure_traces_all_measured_repeats(monkeypatch):
    """Text measure should trace all measured repeats and exclude warmup."""
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    events: list[str] = []
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=1)))

    class _FakeTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure(self, **kwargs) -> SingleMeasurement:
            assert kwargs["trace_path"] is None
            events.append(str(kwargs["progress_desc"]))
            return _trace_single(kwargs["num_prefill"], kwargs["num_decode"])

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_phase_trackers", lambda args, pipeline: (None, None))
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(tps_cli, "_start_qbruntime_trace", lambda path: events.append(f"start:{path}") or object())
    monkeypatch.setattr(tps_cli, "_stop_qbruntime_trace", lambda handle: events.append("stop"))
    monkeypatch.setattr(benchmark_utils, "TPSMeasurer", _FakeTPSMeasurer)

    assert tps_cli._run_text_measure(_trace_scope_args()) == 0
    assert events.count("start:trace.json") == 1
    assert events.count("stop") == 1
    assert events == [
        "warmup generate 1/1",
        "start:trace.json",
        "measure generate 1/2",
        "measure generate 2/2",
        "stop",
    ]


def test_run_text_sweep_traces_all_measured_repeats(monkeypatch):
    """Text sweep should trace all measured repeats and exclude warmup."""
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    events: list[str] = []
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=1)))

    class _FakeTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure(self, **kwargs) -> SingleMeasurement:
            events.append(str(kwargs["progress_desc"]))
            return _trace_single(kwargs["num_prefill"], kwargs["num_decode"])

        def measure_full(self, **kwargs) -> BenchmarkResult:
            assert kwargs["trace_path"] is None
            events.append(str(kwargs["progress_prefix"]))
            return _trace_result()

        def plot_and_save(self, result, save_path) -> None:
            del result, save_path

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_phase_trackers", lambda args, pipeline: (None, None))
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(tps_cli, "_start_qbruntime_trace", lambda path: events.append(f"start:{path}") or object())
    monkeypatch.setattr(tps_cli, "_stop_qbruntime_trace", lambda handle: events.append("stop"))
    monkeypatch.setattr(benchmark_utils, "TPSMeasurer", _FakeTPSMeasurer)

    assert tps_cli._run_text_sweep(_trace_scope_args()) == 0
    assert events.count("start:trace.json") == 1
    assert events.count("stop") == 1
    assert events == ["warmup generate 1/1", "start:trace.json", "run 1/2", "run 2/2", "stop"]


def test_run_vlm_measure_traces_all_measured_repeats(monkeypatch):
    """VLM measure should trace measured vision and LLM phases separately after warmup."""
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    events: list[str] = []
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=1)))

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure_vision(self, **kwargs) -> list[tuple[float, float]]:
            del kwargs
            events.append("vision")
            return [(1.0, 1.0)]

        def measure_llm_full(self, **kwargs) -> BenchmarkResult:
            del kwargs
            events.append("llm")
            return _trace_result(prefill=8, decode=8)

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: None)
    monkeypatch.setattr(tps_cli, "_build_phase_trackers", lambda args, pipeline: (None, None))
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(tps_cli, "_start_qbruntime_trace", lambda path: events.append(f"start:{path}") or object())
    monkeypatch.setattr(tps_cli, "_stop_qbruntime_trace", lambda handle: events.append("stop"))
    monkeypatch.setattr(benchmark_utils, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)

    assert tps_cli._run_vlm_measure(_trace_scope_args(task="image-text-to-text")) == 0
    assert events.count("start:trace.vision.json") == 1
    assert events.count("start:trace.llm.json") == 1
    assert events.count("stop") == 2
    assert events == [
        "vision",
        "start:trace.vision.json",
        "vision",
        "vision",
        "stop",
        "llm",
        "start:trace.llm.json",
        "llm",
        "llm",
        "stop",
    ]


def test_run_vlm_sweep_traces_all_measured_repeats(monkeypatch):
    """VLM sweep should trace measured vision and LLM phases separately after warmup."""
    import mblt_model_zoo.hf_transformers.utils.benchmark_utils as benchmark_utils

    events: list[str] = []
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=1)))

    class _FakeVLMTPSMeasurer:
        def __init__(self, pipeline_arg) -> None:
            assert pipeline_arg is pipeline

        def measure_vision(self, **kwargs) -> list[tuple[float, float]]:
            events.append(f"vision:{kwargs['image_resolution']}")
            return [(1.0, 1.0)]

        def measure_llm_full(self, **kwargs) -> BenchmarkResult:
            events.append(f"llm:{kwargs['image_resolution']}")
            return _trace_result()

    monkeypatch.setattr(tps_cli, "_build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(tps_cli, "_build_device_tracker", lambda args, pipeline: None)
    monkeypatch.setattr(tps_cli, "_build_phase_trackers", lambda args, pipeline: (None, None))
    monkeypatch.setattr(tps_cli, "_print_device_status", lambda args, tracker: None)
    monkeypatch.setattr(tps_cli, "_start_qbruntime_trace", lambda path: events.append(f"start:{path}") or object())
    monkeypatch.setattr(tps_cli, "_stop_qbruntime_trace", lambda handle: events.append("stop"))
    monkeypatch.setattr(benchmark_utils, "VLMTPSMeasurer", _FakeVLMTPSMeasurer)

    args = _trace_scope_args(task="image-text-to-text", image_resolutions=[224, 448], llm_resolution=224)
    assert tps_cli._run_vlm_sweep(args) == 0
    assert events.count("start:trace.vision.json") == 1
    assert events.count("start:trace.llm.json") == 1
    assert events.count("stop") == 2
    assert events == [
        "vision:224",
        "vision:448",
        "start:trace.vision.json",
        "vision:224",
        "vision:224",
        "vision:448",
        "vision:448",
        "stop",
        "llm:224",
        "start:trace.llm.json",
        "llm:224",
        "llm:224",
        "stop",
    ]
