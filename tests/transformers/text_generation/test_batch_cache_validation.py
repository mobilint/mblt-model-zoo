import pytest
import torch

from mblt_model_zoo.hf_transformers.utils import multi_slot_dispatch
from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin


class _FakeCache:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size


class _FakeInputBufferInfo:
    def __init__(self, max_width: int):
        self.max_width = max_width


class _FakeMxqModel:
    _VOCAB_SIZE = 5

    def __init__(self):
        self.calls: list[list[int]] = []

    def get_input_buffer_info(self):
        return [_FakeInputBufferInfo(max_width=2)]

    def get_model_output_shape(self):
        # Static last-only layout: token axis is 1, vocab is the compiled last dim.
        # Batched Path 1 no longer probes this — vocab comes from the raw
        # ndarray's trailing axis — but Path 2/3 and the empty-selector
        # helpers still call ``_mxq_static_vocab_size`` so the method stays.
        return [(1, 1, self._VOCAB_SIZE)]

    def infer(self, inputs, _, __, batch_params):
        self.calls.append([param.cache_id for param in batch_params])
        active_batch = len(batch_params)
        logits = torch.arange(active_batch * self._VOCAB_SIZE, dtype=torch.float32).reshape(
            active_batch, self._VOCAB_SIZE
        )
        return [logits.numpy()]


class _FakeBackend:
    def __init__(self, mxq_model: _FakeMxqModel):
        self.mxq_model = mxq_model
        # ``_llm_forward_batch`` now consults ``npu_backend.mxq_models``
        # directly so per-group dispatch can address specific Model slots.
        # A single-Model fake still exercises the fast path.
        self.mxq_models = [mxq_model]
        self.k_per_model = 1
        self._output_layout_cached = None
        self._dispatcher = None

    @property
    def output_layout(self):
        cached = self._output_layout_cached
        if cached is not None:
            return cached
        try:
            shapes = self.mxq_models[0].get_model_output_shape()
        except Exception:
            return None
        if not shapes:
            return None
        first_shape = tuple(shapes[0])
        if len(first_shape) < 2:
            return None
        token_axis = int(first_shape[-2])
        return "n_tokens" if token_axis == -1 else "n_items"

    def _set_output_layout(self, layout):
        self._output_layout_cached = layout

    @property
    def dispatcher(self):
        if self._dispatcher is None:
            self._dispatcher = multi_slot_dispatch.MultiSlotDispatcher(self)
        return self._dispatcher


def test_validate_batch_cache_accepts_matching_size():
    MobilintModelMixin._validate_batch_cache(_FakeCache(batch_size=4), batch_size=4)


def test_validate_batch_cache_accepts_larger_size():
    MobilintModelMixin._validate_batch_cache(_FakeCache(batch_size=8), batch_size=4)


def test_validate_batch_cache_rejects_smaller_size():
    with pytest.raises(ValueError, match="Batch cache size is too small"):
        MobilintModelMixin._validate_batch_cache(_FakeCache(batch_size=1), batch_size=4)


def test_llm_forward_batch_updates_npu_time_and_preserves_tensor_attributes(monkeypatch: pytest.MonkeyPatch):
    perf_counter_values = iter([1.0, 1.2, 2.0, 2.3])
    # NPU timing accounting lives inside ``MultiSlotDispatcher.dispatch`` — the
    # ``perf_counter`` samples that feed ``npu_time`` are read from
    # ``multi_slot_dispatch.time``, not ``modeling_utils.time``.
    monkeypatch.setattr(multi_slot_dispatch.time, "perf_counter", lambda: next(perf_counter_values))

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


def test_resolve_batched_attention_mask_creates_all_ones_for_batched_models():
    model = MobilintModelMixin.__new__(MobilintModelMixin)
    model.config = type("Config", (), {"max_batch_size": 16})()

    inputs_embeds = torch.randn(1, 3, 4, dtype=torch.float16)

    attention_mask = model.resolve_batched_attention_mask(inputs_embeds, attention_mask=None)

    assert attention_mask is not None
    assert torch.equal(attention_mask, torch.ones((1, 3), dtype=torch.long, device=inputs_embeds.device))


def test_resolve_batched_attention_mask_keeps_none_for_non_batched_models():
    model = MobilintModelMixin.__new__(MobilintModelMixin)
    model.config = type("Config", (), {"max_batch_size": 1})()

    inputs_embeds = torch.randn(1, 3, 4, dtype=torch.float16)

    attention_mask = model.resolve_batched_attention_mask(inputs_embeds, attention_mask=None)

    assert attention_mask is None
