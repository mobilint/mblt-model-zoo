import inspect
import importlib

import pytest
import torch
from transformers import GenerationConfig, GenerationMixin

from mblt_model_zoo.cli import tps as tps_cli
from mblt_model_zoo.cli.main import build_parser
from mblt_model_zoo.hf_transformers.models.blip.modeling_blip_text import MobilintBlipTextLMHeadModel
from mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl import MobilintQwen2VLForConditionalGeneration
from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import MobilintQwen3VLForConditionalGeneration
from mblt_model_zoo.hf_transformers.utils.generation_utils import MobilintGenerationMixin
from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
    BenchmarkResult,
    SingleMeasurement,
    TPSMeasurer,
    VLMTPSMeasurer,
    _resolve_config_vocab_size,
    _resolve_image_features_tensor,
    _supports_fake_decode_prefill,
    _get_npu_timing_target,
    _temporarily_sanitize_generation_config,
    npu_latency_pct,
)


class _DummyConfig:
    def __init__(self, vocab_size: int | None = None, text_config=None):
        if vocab_size is not None:
            self.vocab_size = vocab_size
        if text_config is not None:
            self.text_config = text_config


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
        """Record generation kwargs and stop the streamer without emitting tokens."""
        self.generate_kwargs = kwargs
        self.generation_config_during_generate = {
            "do_sample": self.generation_config.do_sample,
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
            "top_k": self.generation_config.top_k,
        }
        kwargs["streamer"].end()
        return kwargs["input_ids"]


class _DummyTextPipeline:
    """Minimal text-generation pipeline for TPSMeasurer tests."""

    def __init__(self, model: _DummyGenerateNPUModel) -> None:
        self.model = model
        self.tokenizer = _DummyTokenizer()


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


class _DummyVLMTPSMeasurer(VLMTPSMeasurer):
    """VLM TPS measurer double that bypasses pipeline construction."""

    def __init__(self, language_model: _DummyVLMLanguageModel) -> None:
        self.model = _DummyVLMModel(language_model)
        self.tokenizer = object()
        self.processor = object()

    def _get_language_model(self) -> _DummyVLMLanguageModel:
        """Return the dummy language model under test."""
        return self.model.language_model


class _RoutingTPSMeasurer(TPSMeasurer):
    """TPS measurer test double that records real/fake decode routing."""

    def __init__(self, model) -> None:
        self.model = model
        self.calls: list[tuple[str, int, int]] = []

    def measure(
        self,
        num_prefill=512,
        num_decode=128,
        prefill_chunk_size=None,
        trace_path=None,
        show_progress: bool = False,
        progress_desc=None,
        on_prefill_start=None,
        on_prefill_end=None,
        on_decode_start=None,
        on_decode_end=None,
    ) -> SingleMeasurement:
        """Record real-prefill measurements without running generation."""
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
    ) -> SingleMeasurement:
        """Record fake-prefill decode measurements without running generation."""
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


def test_cli_tps_sweep_range_parsing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "sweep",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--prefill-range",
            "1:3:1",
            "--cache-lengths",
            "1024,2048,4096",
            "--no-plot",
        ]
    )
    assert args.prefill_range == (1, 3, 1)
    assert args.cache_lengths == [1024, 2048, 4096]
    assert args.plot is None
    assert args.device_backend is None


def test_cli_tps_measure_defaults():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
        ]
    )

    assert args.prefill == 128
    assert args.decode == 32


def test_cli_tps_sweep_defaults():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "sweep",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
        ]
    )

    assert args.prefill_range == (512, 2048, 512)
    assert args.cache_lengths == [128, 512, 1024, 2048]
    assert args.decode_window == 32


def test_cli_tps_sweep_vlm_options_parsing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "sweep",
            "--model",
            "mobilint/Qwen2-VL-2B-Instruct",
            "--task",
            "image-text-to-text",
            "--prefill-range",
            "512:2048:512",
            "--cache-lengths",
            "128,512,1024,2048",
            "--decode-window",
            "32",
            "--image-resolutions",
            "224,448",
            "--llm-resolution",
            "224",
            "--prompt",
            "Describe.",
            "--no-plot",
        ]
    )

    assert args.prefill_range == (512, 2048, 512)
    assert args.cache_lengths == [128, 512, 1024, 2048]
    assert args.decode_window == 32
    assert args.image_resolutions == [224, 448]
    assert args.llm_resolution == 224
    assert args.prompt == "Describe."
    assert args.plot is None


def test_cli_tps_vlm_sweep_removed():
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["tps", "vlm-sweep", "--model", "dummy"])

    assert excinfo.value.code == 2


def test_cmd_sweep_routes_vlm_task(monkeypatch):
    calls: list[str] = []

    def fake_vlm_sweep(args):
        calls.append(f"vlm:{args.task}")
        return 0

    def fake_text_sweep(args):
        calls.append(f"text:{args.task}")
        return 0

    monkeypatch.setattr(tps_cli, "_run_vlm_sweep", fake_vlm_sweep)
    monkeypatch.setattr(tps_cli, "_run_text_sweep", fake_text_sweep)

    assert tps_cli._cmd_sweep(type("Args", (), {"task": "image-text-to-text"})()) == 0
    assert tps_cli._cmd_sweep(type("Args", (), {"task": "text-generation"})()) == 0
    assert calls == ["vlm:image-text-to-text", "text:text-generation"]


def test_cmd_measure_routes_vlm_task(monkeypatch):
    calls: list[str] = []

    def fake_vlm_measure(args):
        calls.append(f"vlm:{args.task}")
        return 0

    def fake_text_measure(args):
        calls.append(f"text:{args.task}")
        return 0

    monkeypatch.setattr(tps_cli, "_run_vlm_measure", fake_vlm_measure)
    monkeypatch.setattr(tps_cli, "_run_text_measure", fake_text_measure)

    assert tps_cli._cmd_measure(type("Args", (), {"task": "image-text-to-text"})()) == 0
    assert tps_cli._cmd_measure(type("Args", (), {"task": "text-generation"})()) == 0
    assert calls == ["vlm:image-text-to-text", "text:text-generation"]


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


def test_resolve_image_features_tensor_requires_tensor():
    with pytest.raises(TypeError, match="image feature tensor"):
        _resolve_image_features_tensor(_DummyVisionOutput())


@pytest.mark.parametrize(
    ("model_cls", "method_name"),
    [
        (MobilintQwen2VLForConditionalGeneration, "forward"),
        (MobilintQwen2VLForConditionalGeneration, "prepare_inputs_for_generation"),
        (MobilintQwen3VLForConditionalGeneration, "forward"),
        (MobilintQwen3VLForConditionalGeneration, "prepare_inputs_for_generation"),
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
        "mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl:MobilintQwen3VLForConditionalGeneration",
        "mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl:MobilintQwen3VLTextModel",
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
        prefill_chunk_size=64,
    )

    assert model_inputs["count_npu_time"] is True
    assert model_inputs["prefill_chunk_size"] == 64


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
