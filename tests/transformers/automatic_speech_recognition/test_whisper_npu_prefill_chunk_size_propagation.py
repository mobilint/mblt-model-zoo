"""Regression tests for npu_prefill_chunk_size propagation through the Whisper stack.

Verifies:
- ``decoder_forward`` honors ``npu_prefill_chunk_size`` by chunking long token suffixes into
  multiple ``mxq_model.infer`` calls with monotonically advancing ``prefix_length``.
- The outer forward chain (``MobilintWhisperForConditionalGeneration.forward`` →
  ``MobilintWhisperModel.forward`` → ``MobilintWhisperDecoder.forward``) threads the value
  down to ``decoder_forward`` rather than silently dropping it in ``**kwargs``.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper import (
    MobilintWhisperDecoder,
    MobilintWhisperForConditionalGeneration,
    MobilintWhisperModel,
)
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintWhisperCache


class _ChunkTrackingMxqModel:
    """MXQ stub that records each ``infer`` call's suffix length and ``prefix_length``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []

    def infer(self, inputs: list[object], output_buffer: object, cache_size: int) -> list[np.ndarray]:
        del output_buffer
        hidden_states, _encoder_hidden_states = inputs
        suffix_length = int(hidden_states.shape[2])
        self.calls.append({"suffix_length": suffix_length, "prefix_length": int(cache_size)})
        return [np.zeros((suffix_length, 4), dtype=np.float32)]


def _make_decoder(mxq_model: _ChunkTrackingMxqModel) -> MobilintWhisperDecoder:
    decoder = object.__new__(MobilintWhisperDecoder)
    decoder.npu_backend = SimpleNamespace(mxq_model=mxq_model)

    def embed_token_suffix(
        self: MobilintWhisperDecoder,
        token_history: list[int],
        start_index: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        del self
        suffix_length = len(token_history) - start_index
        return torch.ones((1, 1, suffix_length, 2), dtype=torch.float32, device=device)

    decoder._embed_token_suffix = MethodType(embed_token_suffix, decoder)
    return decoder


def test_decoder_forward_chunks_long_suffix() -> None:
    """A 5-token suffix with npu_prefill_chunk_size=2 should trigger 3 infer calls."""
    mxq_model = _ChunkTrackingMxqModel()
    decoder = _make_decoder(mxq_model)
    cache = MobilintWhisperCache(mxq_model, batch_size=1)

    hidden_states = torch.ones((1, 1, 1, 2), dtype=torch.float32)
    encoder_hidden_states = torch.ones((1, 1, 1, 2), dtype=torch.float32)
    input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)

    decoder.decoder_forward(
        hidden_states,
        encoder_hidden_states,
        cache,
        torch.tensor([0], dtype=torch.long),
        input_ids=input_ids,
        npu_prefill_chunk_size=2,
    )

    assert [call["suffix_length"] for call in mxq_model.calls] == [2, 2, 1]
    assert [call["prefix_length"] for call in mxq_model.calls] == [0, 2, 4]


def test_decoder_forward_without_chunk_size_uses_single_infer() -> None:
    """Without npu_prefill_chunk_size (and no config), a 5-token suffix uses a single infer."""
    mxq_model = _ChunkTrackingMxqModel()
    decoder = _make_decoder(mxq_model)
    cache = MobilintWhisperCache(mxq_model, batch_size=1)

    hidden_states = torch.ones((1, 1, 1, 2), dtype=torch.float32)
    encoder_hidden_states = torch.ones((1, 1, 1, 2), dtype=torch.float32)
    input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)

    decoder.decoder_forward(
        hidden_states,
        encoder_hidden_states,
        cache,
        torch.tensor([0], dtype=torch.long),
        input_ids=input_ids,
    )

    assert len(mxq_model.calls) == 1
    assert mxq_model.calls[0]["suffix_length"] == 5


def test_decoder_forward_signature_declares_npu_prefill_chunk_size() -> None:
    """MobilintWhisperDecoder.forward must accept the kwarg so callers below can pass it down."""
    import inspect

    for cls in (
        MobilintWhisperDecoder,
        MobilintWhisperModel,
        MobilintWhisperForConditionalGeneration,
    ):
        params = inspect.signature(cls.forward).parameters
        assert "npu_prefill_chunk_size" in params, (
            f"{cls.__name__}.forward must accept npu_prefill_chunk_size"
        )


def test_decoder_wrapper_forward_threads_npu_prefill_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """MobilintWhisperDecoder.forward should forward npu_prefill_chunk_size to decoder_forward."""
    captured: dict[str, object] = {}

    def fake_decoder_forward(
        self,
        hidden_states,
        encoder_hidden_states,
        past_key_values,
        cache_position,
        *,
        input_ids=None,
        npu_prefill_chunk_size=None,
    ):
        captured["npu_prefill_chunk_size"] = npu_prefill_chunk_size
        return torch.zeros((hidden_states.shape[0], 1, hidden_states.shape[2], 4), dtype=torch.float32)

    monkeypatch.setattr(MobilintWhisperDecoder, "decoder_forward", fake_decoder_forward)

    decoder = object.__new__(MobilintWhisperDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    decoder.embed_tokens = torch.nn.Embedding(4, 3)

    def fake_embed_positions(inputs, past_key_values_length, position_ids):
        del past_key_values_length, position_ids
        if inputs.dtype == torch.long:
            batch, seq = inputs.shape
            return torch.zeros((batch, seq, 3), dtype=torch.float32)
        return torch.zeros_like(inputs)

    decoder.embed_positions = fake_embed_positions

    decoder.forward(
        input_ids=torch.tensor([[0, 1]], dtype=torch.long),
        encoder_hidden_states=torch.ones((1, 1, 2, 3), dtype=torch.float32),
        npu_prefill_chunk_size=64,
    )

    assert captured["npu_prefill_chunk_size"] == 64


def test_model_forward_threads_npu_prefill_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """MobilintWhisperModel.forward should forward npu_prefill_chunk_size to the decoder."""
    captured: dict[str, object] = {}

    def fake_decoder_call(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            last_hidden_state=torch.zeros((1, 1, 2, 4), dtype=torch.float32),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    model = object.__new__(MobilintWhisperModel)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    model.decoder = fake_decoder_call

    from transformers.modeling_outputs import BaseModelOutput

    encoder_outputs = BaseModelOutput(
        last_hidden_state=torch.ones((1, 1, 2, 3), dtype=torch.float32),
        hidden_states=None,
        attentions=None,
    )

    model.forward(
        decoder_input_ids=torch.tensor([[0, 1]], dtype=torch.long),
        encoder_outputs=encoder_outputs,
        npu_prefill_chunk_size=128,
    )

    assert captured["npu_prefill_chunk_size"] == 128


def test_conditional_generation_forward_threads_npu_prefill_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """MobilintWhisperForConditionalGeneration.forward should pass npu_prefill_chunk_size to self.model."""
    from transformers.modeling_outputs import Seq2SeqModelOutput

    captured: dict[str, object] = {}

    def fake_model_call(*args, **kwargs):
        del args
        captured.update(kwargs)
        return Seq2SeqModelOutput(
            last_hidden_state=torch.zeros((1, 1, 2, 4), dtype=torch.float32),
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    gen = object.__new__(MobilintWhisperForConditionalGeneration)
    gen.config = SimpleNamespace(return_dict=True, vocab_size=4)
    gen.max_target_positions = 8
    gen.model = fake_model_call

    gen.forward(
        decoder_input_ids=torch.tensor([[0, 1]], dtype=torch.long),
        npu_prefill_chunk_size=256,
    )

    assert captured["npu_prefill_chunk_size"] == 256
