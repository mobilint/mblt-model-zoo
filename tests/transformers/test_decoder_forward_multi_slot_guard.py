"""Regression tests for the encoder-decoder ``decoder_forward`` N=1 guard.

The shared :meth:`MobilintModelMixin.decoder_forward` implementation dispatches
one blocking ``mxq_model.infer`` on slot 0 with the caller-supplied ``[hidden,
encoder_hidden]`` payload. Growing the backend to ``N>1`` slots (e.g. CLI
``--batch-size B`` on a K==1 text MXQ) would leave slots ``1..N-1`` idle while
slot 0 receives a batch-shaped input it cannot serve — and there is no
``reorder_cache`` across slots for beam search either. The centralized guard
should therefore fail with actionable guidance on the first call rather than
producing a cryptic qbruntime shape error.

These tests exercise the guard without booting an NPU by binding
``MobilintModelMixin`` to a lightweight fake backend that exposes a
configurable ``mxq_models`` list.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache
from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin


class _RecordingCrossAttnMxq:
    """Minimal MXQ stub that echoes the encoder_hidden last-token shape as logits."""

    def __init__(self, vocab_size: int = 5) -> None:
        self.vocab_size = vocab_size
        self.calls: list[dict] = []

    def get_cache_infos(self):
        return []

    def infer(self, inputs, _extra, cache_size, batch_params=None):
        hidden_chunk = np.asarray(inputs[0])
        encoder_chunk = np.asarray(inputs[1])
        self.calls.append(
            {
                "hidden_shape": tuple(hidden_chunk.shape),
                "encoder_shape": tuple(encoder_chunk.shape),
                "cache_size": int(cache_size),
                "batch_params": batch_params,
            }
        )
        # (batch=1, 1, seq=1, vocab) — encoder-decoder base returns
        # per-token logits; the exact values don't matter for guard tests.
        seq_axis = hidden_chunk.shape[-2] if hidden_chunk.ndim >= 2 else 1
        return [
            np.zeros((1, 1, seq_axis, self.vocab_size), dtype=np.float32),
        ]


class _MultiSlotFakeBackend:
    """FakeBackend exposing ``N`` fake ``qbruntime.Model`` slots."""

    def __init__(self, mxq_models: List[_RecordingCrossAttnMxq]) -> None:
        self.mxq_models = list(mxq_models)
        self.mxq_model = self.mxq_models[0]
        self.k_per_model = 1
        self._output_layout_cached = None
        self._dispatcher = None

    @property
    def output_layout(self):
        return self._output_layout_cached

    def _set_output_layout(self, layout):
        self._output_layout_cached = layout

    @property
    def dispatcher(self):
        if self._dispatcher is None:
            from mblt_model_zoo.hf_transformers.utils.multi_slot_dispatch import MultiSlotDispatcher

            self._dispatcher = MultiSlotDispatcher(self)
        return self._dispatcher


def _make_encoder_decoder_model(mxq_models: List[_RecordingCrossAttnMxq]) -> MobilintModelMixin:
    """Bind a bare :class:`MobilintModelMixin` to a fake multi-slot backend."""
    model = MobilintModelMixin.__new__(MobilintModelMixin)
    model.npu_backend = _MultiSlotFakeBackend(mxq_models)
    model.config = type(
        "Config",
        (),
        {"npu_prefill_chunk_size": None, "max_batch_size": len(mxq_models)},
    )()
    model.npu_time = None
    return model


def test_decoder_forward_n1_slot_passes_through() -> None:
    """N==1: legacy single-slot dispatch stays intact and reaches slot 0's ``.infer``."""
    m0 = _RecordingCrossAttnMxq(vocab_size=3)
    model = _make_encoder_decoder_model([m0])
    cache = MobilintCache(m0, per_model_batch=1)

    hidden = torch.zeros(1, 1, 1, 4, dtype=torch.float32)
    encoder_hidden = torch.zeros(1, 1, 2, 4, dtype=torch.float32)
    cache_position = torch.arange(1)

    logits = model.decoder_forward(hidden, encoder_hidden, cache, cache_position)

    assert len(m0.calls) == 1
    # No batch_params on the encoder-decoder single-slot path.
    assert m0.calls[0]["batch_params"] is None
    assert logits.shape[-1] == 3


def test_decoder_forward_multi_slot_backend_raises_clear_error() -> None:
    """N>1: the guard raises with actionable guidance and no ``.infer`` is dispatched."""
    m0 = _RecordingCrossAttnMxq()
    m1 = _RecordingCrossAttnMxq()
    model = _make_encoder_decoder_model([m0, m1])
    cache = MobilintCache([m0, m1], per_model_batch=1)

    hidden = torch.zeros(2, 1, 1, 4, dtype=torch.float32)
    encoder_hidden = torch.zeros(2, 1, 2, 4, dtype=torch.float32)
    cache_position = torch.arange(1)

    with pytest.raises(NotImplementedError) as excinfo:
        model.decoder_forward(hidden, encoder_hidden, cache, cache_position)

    message = str(excinfo.value)
    # The message must name the launched slot count and point at the two
    # remediations so users can act on it without spelunking through code.
    assert "N=2" in message
    assert "--batch-size" in message
    assert "K>1" in message

    # No ``.infer`` should have been dispatched — the guard fails before any
    # NPU work happens.
    assert m0.calls == []
    assert m1.calls == []
