"""Shared fake MXQ backends for the LLM-core ``llm_forward`` tests.

These stubs stand in for a real MXQ compiled model so tests can exercise the
shared ``MobilintModelMixin.llm_forward`` code paths (fast/last-only, dynamic
axis all-logits, and last-only fallback) without booting an NPU.

The Qwen3-VL / Qwen2-VL text decoders have their own dual-input variants of
these fakes (with a different ``infer`` signature); those live next to the
respective VL tests and are intentionally not unified here.
"""

from __future__ import annotations

import numpy as np

from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin


class FakeInputBufferInfo:
    def __init__(self, max_width: int, max_cache_size: int = 128):
        self.max_width = max_width
        self.max_cache_size = max_cache_size


class StaticLastOnlyMxq:
    """MXQ stub that emits only the last-token logits per infer call.

    ``infer`` accepts both the single-item shape (``[chunk], None, cache_size``)
    and the batched shape (``[chunk], None, 0, batch_params``). Every call is
    recorded so tests can assert chunk boundaries.
    """

    def __init__(self, vocab_size: int = 5, max_width: int = 4):
        self.vocab_size = vocab_size
        self.max_width = max_width
        self.calls: list[dict] = []
        self._batched_counter = 0

    def get_input_buffer_info(self):
        return [FakeInputBufferInfo(max_width=self.max_width)]

    def get_model_output_shape(self):
        # Static token axis (second-to-last); "1" signals last-token-only output.
        return [(1, 1, self.vocab_size)]

    def infer(self, inputs, _extra, cache_size, batch_params=None):
        chunk = np.asarray(inputs[0])
        if batch_params is None:
            self.calls.append({"shape": tuple(chunk.shape), "cache_size": int(cache_size)})
            return [np.full((1, 1, self.vocab_size), fill_value=float(cache_size), dtype=np.float32)]

        self._batched_counter += 1
        active_batch = len(batch_params)
        payload = np.stack(
            [
                np.full(
                    (self.vocab_size,), fill_value=float(p.cache_id * 100 + self._batched_counter), dtype=np.float32
                )
                for p in batch_params
            ],
            axis=0,
        )
        self.calls.append(
            {
                "shape": tuple(chunk.shape),
                "batch": [(p.cache_id, p.sequence_length, p.cache_size) for p in batch_params],
            }
        )
        assert payload.shape == (active_batch, self.vocab_size)
        return [payload]


class DynamicAxisMxq:
    """MXQ stub that emits per-position logits."""

    def __init__(self, vocab_size: int = 5, max_width: int = 4):
        self.vocab_size = vocab_size
        self.max_width = max_width
        self.calls: list[dict] = []
        self._batched_counter = 0

    def get_input_buffer_info(self):
        return [FakeInputBufferInfo(max_width=self.max_width)]

    def get_model_output_shape(self):
        # Dynamic token axis; the shared helper interprets this as all-logits.
        return [(1, -1, self.vocab_size)]

    def infer(self, inputs, _extra, cache_size, batch_params=None):
        chunk = np.asarray(inputs[0])
        if batch_params is None:
            # Single-item: chunk is (1, 1, chunk_len, hidden). Emit
            # (1, chunk_len, vocab) so the token axis is second-to-last.
            chunk_len = int(chunk.shape[2])
            offset = int(cache_size) * self.vocab_size
            values = np.arange(chunk_len * self.vocab_size, dtype=np.float32) + offset
            self.calls.append({"shape": tuple(chunk.shape), "cache_size": int(cache_size), "chunk_len": chunk_len})
            return [values.reshape(1, chunk_len, self.vocab_size)]

        self._batched_counter += 1
        total_tokens = sum(int(p.sequence_length) for p in batch_params)
        flat = np.arange(total_tokens * self.vocab_size, dtype=np.float32).reshape(total_tokens, self.vocab_size)
        # Encode per-item id into the payload so batched tests can spot mixups.
        offset = 0
        for p in batch_params:
            flat[offset : offset + int(p.sequence_length), :] += p.cache_id * 1000.0
            offset += int(p.sequence_length)
        self.calls.append(
            {
                "shape": tuple(chunk.shape),
                "batch": [(p.cache_id, p.sequence_length, p.cache_size) for p in batch_params],
            }
        )
        return [flat]


class FakeBackend:
    def __init__(self, mxq_model, *, n_slots: int = 1, k_per_model: int = 1):
        self.mxq_model = mxq_model
        # Static-last-only Batch<K> MXQs compile a real batch axis K and a
        # single Model handles the whole batch; dynamic-axis MXQs are Batch1
        # per Model and multi-slot dispatch scales across ``n_slots`` copies.
        # ``make_model`` picks the parametrization that matches the MXQ
        # variant so ``N * K >= max_batch_size`` clears the cacheless
        # dispatch guard without perturbing the path-selection heuristics.
        self.mxq_models = [mxq_model] * n_slots
        self.k_per_model = k_per_model
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
            from mblt_model_zoo.hf_transformers.utils.multi_slot_dispatch import MultiSlotDispatcher

            self._dispatcher = MultiSlotDispatcher(self)
        return self._dispatcher


def make_model(mxq, *, max_batch_size: int = 1) -> MobilintModelMixin:
    """Construct a bare ``MobilintModelMixin`` without triggering NPU init.

    ``max_batch_size`` scales the fake backend so ``N * K >= max_batch_size``
    and the cacheless dispatch guard (``n_items <= N * K``) clears. The
    scaling target matches how a real backend would handle ``max_batch_size``
    for each MXQ variant:

    * Static-last-only MXQs are compiled with a real batch axis K, so a
      Batch<N> model runs as a single Model with ``k_per_model=N``. The
      call-sequence assertions in :class:`TestLlmForwardBatch` rely on the
      single-slot dispatch that this parametrization produces.
    * Dynamic-axis MXQs are Batch1 per Model in production; multi-slot
      dispatch spreads a Batch<N> request across ``n_slots=N`` copies with
      ``k_per_model=1`` so the K-aware probe still classifies the MXQ as
      Path-2 eligible.
    """
    model = MobilintModelMixin.__new__(MobilintModelMixin)
    scale = max(max_batch_size, 1)
    if isinstance(mxq, StaticLastOnlyMxq):
        model.npu_backend = FakeBackend(mxq, k_per_model=scale)
    else:
        model.npu_backend = FakeBackend(mxq, n_slots=scale)
    config_kwargs = {"npu_prefill_chunk_size": None}
    if max_batch_size > 1:
        config_kwargs["max_batch_size"] = max_batch_size
    model.config = type("Config", (), config_kwargs)()
    model.npu_time = None
    return model
