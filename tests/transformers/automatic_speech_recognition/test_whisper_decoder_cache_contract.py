"""Regression tests for Whisper decoder cache reuse contracts."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import torch

from mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper import MobilintWhisperDecoder
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintWhisperCache


class _EncoderAwareMxqModel:
    """MXQ stub that makes decoder logits depend on the supplied encoder row."""

    def __init__(self) -> None:
        self.infer_calls = 0

    def infer(self, inputs: list[object], output_buffer: object, cache_size: int) -> list[np.ndarray]:
        """Return logits whose value identifies the encoder input row."""
        del output_buffer, cache_size
        hidden_states, encoder_hidden_states = inputs
        self.infer_calls += 1
        suffix_length = int(hidden_states.shape[2])
        encoder_value = float(encoder_hidden_states.mean())
        return [np.full((suffix_length, 4), encoder_value, dtype=np.float32)]


def _make_decoder(mxq_model: _EncoderAwareMxqModel) -> MobilintWhisperDecoder:
    """Create a minimal decoder instance for exercising ``decoder_forward`` directly."""
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


def test_whisper_duplicate_logits_are_not_reused_across_audio_rows() -> None:
    """Identical decoder tokens from different audio rows should use row-specific encoder states."""
    mxq_model = _EncoderAwareMxqModel()
    decoder = _make_decoder(mxq_model)
    cache = MobilintWhisperCache(mxq_model, batch_size=2)
    hidden_states = torch.ones((2, 1, 1, 2), dtype=torch.float32)
    encoder_hidden_states = torch.tensor(
        [
            [[[1.0, 1.0]]],
            [[[7.0, 7.0]]],
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[42], [42]], dtype=torch.long)

    logits = decoder.decoder_forward(
        hidden_states,
        encoder_hidden_states,
        cache,
        torch.tensor([0], dtype=torch.long),
        input_ids=input_ids,
    )

    assert mxq_model.infer_calls == 2
    assert logits.shape == (2, 1, 1, 4)
    assert torch.equal(logits[0], torch.full((1, 1, 4), 1.0))
    assert torch.equal(logits[1], torch.full((1, 1, 4), 7.0))


def test_whisper_duplicate_logits_still_reuse_single_source_beams() -> None:
    """Beam-expanded rows from one encoder source should keep the duplicate-logit shortcut."""
    mxq_model = _EncoderAwareMxqModel()
    decoder = _make_decoder(mxq_model)
    cache = MobilintWhisperCache(mxq_model, batch_size=2)
    hidden_states = torch.ones((2, 1, 1, 2), dtype=torch.float32)
    encoder_hidden_states = torch.tensor([[[[3.0, 3.0]]]], dtype=torch.float32)
    input_ids = torch.tensor([[42], [42]], dtype=torch.long)

    logits = decoder.decoder_forward(
        hidden_states,
        encoder_hidden_states,
        cache,
        torch.tensor([0], dtype=torch.long),
        input_ids=input_ids,
    )

    assert mxq_model.infer_calls == 1
    assert logits.shape == (2, 1, 1, 4)
    assert torch.equal(logits[0], logits[1])


def test_whisper_duplicate_logits_reuse_beam_expanded_single_source_rows() -> None:
    """Beam-expanded encoder rows for one audio source should share the same source id."""
    mxq_model = _EncoderAwareMxqModel()
    decoder = _make_decoder(mxq_model)
    cache = MobilintWhisperCache(mxq_model, batch_size=2)
    cache.set_encoder_source_count(1)
    hidden_states = torch.ones((2, 1, 1, 2), dtype=torch.float32)
    encoder_hidden_states = torch.tensor(
        [
            [[[3.0, 3.0]]],
            [[[3.0, 3.0]]],
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[42], [42]], dtype=torch.long)

    logits = decoder.decoder_forward(
        hidden_states,
        encoder_hidden_states,
        cache,
        torch.tensor([0], dtype=torch.long),
        input_ids=input_ids,
    )

    assert mxq_model.infer_calls == 1
    assert logits.shape == (2, 1, 1, 4)
    assert torch.equal(logits[0], logits[1])


def test_whisper_duplicate_logits_reuse_only_within_beam_expanded_source_groups() -> None:
    """Beam grouping should prevent duplicate logits from crossing original audio sources."""
    mxq_model = _EncoderAwareMxqModel()
    decoder = _make_decoder(mxq_model)
    cache = MobilintWhisperCache(mxq_model, batch_size=4)
    cache.set_encoder_source_count(2)
    hidden_states = torch.ones((4, 1, 1, 2), dtype=torch.float32)
    encoder_hidden_states = torch.tensor(
        [
            [[[1.0, 1.0]]],
            [[[1.0, 1.0]]],
            [[[7.0, 7.0]]],
            [[[7.0, 7.0]]],
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[42], [42], [42], [42]], dtype=torch.long)

    logits = decoder.decoder_forward(
        hidden_states,
        encoder_hidden_states,
        cache,
        torch.tensor([0], dtype=torch.long),
        input_ids=input_ids,
    )

    assert mxq_model.infer_calls == 2
    assert logits.shape == (4, 1, 1, 4)
    assert torch.equal(logits[0], logits[1])
    assert torch.equal(logits[2], logits[3])
    assert not torch.equal(logits[0], logits[2])
