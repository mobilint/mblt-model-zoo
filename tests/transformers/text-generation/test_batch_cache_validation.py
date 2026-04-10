import pytest
import torch

from mblt_model_zoo.hf_transformers.utils import modeling_utils
from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin


class _FakeCache:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size


class _FakeInputBufferInfo:
    def __init__(self, max_width: int):
        self.max_width = max_width


class _FakeMxqModel:
    def __init__(self):
        self.calls: list[list[int]] = []

    def get_input_buffer_info(self):
        return [_FakeInputBufferInfo(max_width=2)]

    def infer(self, inputs, _, __, batch_params):
        self.calls.append([param.cache_id for param in batch_params])
        active_batch = len(batch_params)
        vocab_size = 5
        logits = torch.arange(active_batch * vocab_size, dtype=torch.float32).reshape(active_batch, vocab_size)
        return [logits.numpy()]


class _FakeBackend:
    def __init__(self, mxq_model: _FakeMxqModel):
        self.mxq_model = mxq_model


def test_validate_batch_cache_accepts_matching_size():
    MobilintModelMixin._validate_batch_cache(_FakeCache(batch_size=4), batch_size=4)


def test_validate_batch_cache_accepts_larger_size():
    MobilintModelMixin._validate_batch_cache(_FakeCache(batch_size=8), batch_size=4)


def test_validate_batch_cache_rejects_smaller_size():
    with pytest.raises(ValueError, match="Batch cache size is too small"):
        MobilintModelMixin._validate_batch_cache(_FakeCache(batch_size=1), batch_size=4)


def test_llm_forward_batch_updates_npu_time_and_preserves_tensor_attributes(monkeypatch: pytest.MonkeyPatch):
    perf_counter_values = iter([1.0, 1.2, 2.0, 2.3])
    monkeypatch.setattr(modeling_utils.time, "perf_counter", lambda: next(perf_counter_values))

    model = MobilintModelMixin.__new__(MobilintModelMixin)
    model.npu_backend = _FakeBackend(_FakeMxqModel())
    model.config = type("Config", (), {"npu_prefill_chunk_size": 2})()
    model.npu_time = None

    inputs_embeds = torch.randn(2, 3, 4, dtype=torch.float16)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    cache_position = torch.arange(inputs_embeds.shape[1])

    logits = model.llm_forward(
        inputs_embeds=inputs_embeds,
        past_key_values=None,
        cache_position=cache_position,
        count_npu_time=True,
        attention_mask=attention_mask,
    )

    assert model.npu_time == pytest.approx(0.5)
    assert logits.dtype == inputs_embeds.dtype
    assert logits.device == inputs_embeds.device
    assert logits.shape == (2, 1, 5)


def test_llm_forward_batch_rejects_zero_length_rows():
    model = MobilintModelMixin.__new__(MobilintModelMixin)
    model.npu_backend = _FakeBackend(_FakeMxqModel())
    model.config = type("Config", (), {"npu_prefill_chunk_size": 2})()
    model.npu_time = None

    inputs_embeds = torch.randn(2, 3, 4, dtype=torch.float16)
    attention_mask = torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.long)
    cache_position = torch.arange(inputs_embeds.shape[1])

    with pytest.raises(ValueError, match="Zero-length rows: \\[0\\]"):
        model.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            cache_position=cache_position,
            count_npu_time=False,
            attention_mask=attention_mask,
        )
