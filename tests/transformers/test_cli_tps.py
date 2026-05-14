import inspect

import pytest
import torch

from mblt_model_zoo.cli.main import build_parser
from mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl import MobilintQwen2VLForConditionalGeneration
from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import MobilintQwen3VLForConditionalGeneration
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
    SingleMeasurement,
    TPSMeasurer,
    VLMTPSMeasurer,
    _resolve_config_vocab_size,
    _resolve_image_features_tensor,
    _supports_fake_decode_prefill,
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


class _DummyNonNPUModel:
    """Non-NPU model marker without Mobilint cache support."""


class _DummyVLMContainer:
    """VLM-like container with a nested language model."""

    def __init__(self, language_model) -> None:
        self.model = type("NestedModel", (), {"language_model": language_model})()


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
    assert args.device_backend == "none"


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


def test_tps_measure_full_uses_fake_decode_prefill_for_npu_models():
    measurer = _RoutingTPSMeasurer(_DummyNPUModel())

    result = measurer.measure_full(prefill_range=(1, 1, 1), cache_lengths=[4], decode_window=2)

    assert measurer.calls == [("real", 1, 1), ("fake", 4, 2)]
    assert result.decode_sweep.x_values == [4]
    assert result.decode_sweep.tps_values == [2.0]


def test_tps_measure_full_keeps_real_decode_prefill_for_non_npu_models():
    measurer = _RoutingTPSMeasurer(_DummyNonNPUModel())

    result = measurer.measure_full(prefill_range=(1, 1, 1), cache_lengths=[4], decode_window=2)

    assert measurer.calls == [("real", 1, 1), ("real", 4, 2)]
    assert result.decode_sweep.x_values == [4]
    assert result.decode_sweep.tps_values == [2.0]


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
        (MobilintQwen3VLForConditionalGeneration, "forward"),
        (MobilintQwen3VLForConditionalGeneration, "prepare_inputs_for_generation"),
    ],
)
def test_mobilint_generation_hooks_accept_count_npu_time(model_cls, method_name: str):
    signature = inspect.signature(getattr(model_cls, method_name))

    assert "count_npu_time" in signature.parameters


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
