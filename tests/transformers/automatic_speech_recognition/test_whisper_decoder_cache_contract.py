"""Regression tests for Whisper decoder cache reuse contracts."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper import (
    MobilintWhisperDecoder,
    MobilintWhisperForConditionalGeneration,
)
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


def test_whisper_embeds_only_decoder_does_not_create_default_cache() -> None:
    """Keep embeds-only decoder calls on the non-cache path when cache defaults to enabled."""
    decoder = object.__new__(MobilintWhisperDecoder)
    decoder.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_cache=True,
        return_dict=True,
    )

    def get_mxq_model(self: MobilintWhisperDecoder) -> object:
        del self
        raise AssertionError("embeds-only decoder calls must not create a MobilintWhisperCache")

    def embed_positions(
        inputs: torch.Tensor,
        *,
        past_key_values_length: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        del past_key_values_length, position_ids
        return torch.zeros_like(inputs)

    def decoder_forward(
        self: MobilintWhisperDecoder,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_values: MobilintWhisperCache | None,
        cache_position: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        npu_prefill_chunk_size: int | None = None,
    ) -> torch.Tensor:
        del self, encoder_hidden_states, cache_position, npu_prefill_chunk_size
        assert past_key_values is None
        assert input_ids is None
        return torch.zeros(
            (hidden_states.shape[0], 1, hidden_states.shape[2], 4),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    decoder.get_mxq_model = MethodType(get_mxq_model, decoder)
    decoder.embed_positions = embed_positions
    decoder.decoder_forward = MethodType(decoder_forward, decoder)

    outputs = decoder.forward(
        inputs_embeds=torch.ones((1, 2, 3), dtype=torch.float32),
        encoder_hidden_states=torch.ones((1, 1, 2, 3), dtype=torch.float32),
    )

    assert outputs.past_key_values is None
    assert outputs.last_hidden_state.shape == (1, 1, 2, 4)


# ---------------------------------------------------------------------------
# _get_cache backend-relaunch revalidation
#
# The N==1 backend-topology guard on ``MobilintBeamCache`` / ``MobilintWhisperCache``
# runs only in the constructor. When ``_get_cache`` reuses a cached beam cache
# via ``reset()`` instead of rebuilding, the guard is never re-invoked and can
# silently retain a stale slot handle from a disposed backend. These tests pin
# the fix: the reuse guard must revalidate the live decoder-backend topology on
# every call, rebuild on any change (fresh slot 0 or expanded slot count), and
# preserve the reset fast-path when the topology still matches.
# ---------------------------------------------------------------------------


def _make_whisper_generation_stub(
    mxq_models: list[object], max_batch_size: int = 1
) -> MobilintWhisperForConditionalGeneration:
    """Create a minimal Whisper generation instance for exercising ``_get_cache`` directly."""
    model = object.__new__(MobilintWhisperForConditionalGeneration)
    decoder = SimpleNamespace(
        npu_backend=SimpleNamespace(
            mxq_model=mxq_models[0],
            mxq_models=list(mxq_models),
        ),
    )

    def get_mxq_model(self: object) -> object:
        del self
        return decoder.npu_backend.mxq_model

    decoder.get_mxq_model = MethodType(get_mxq_model, decoder)

    def get_decoder(self: MobilintWhisperForConditionalGeneration) -> object:
        del self
        return decoder

    model.get_decoder = MethodType(get_decoder, model)
    model.config = SimpleNamespace(max_batch_size=max_batch_size)
    return model


def test_whisper_get_cache_resets_when_topology_matches() -> None:
    """Reuse the existing Whisper cache via ``reset()`` when slot 0 is unchanged."""
    slot_0 = object()
    model = _make_whisper_generation_stub(mxq_models=[slot_0], max_batch_size=2)

    first_cache = model._get_cache("mobilint", batch_size=2, max_cache_len=1)
    assert isinstance(first_cache, MobilintWhisperCache)
    assert first_cache.mxq_models == [slot_0]

    second_cache = model._get_cache("mobilint", batch_size=2, max_cache_len=1)
    assert second_cache is first_cache


def test_whisper_get_cache_rebuilds_when_slot_handle_changes() -> None:
    """Reuse must invalidate when the decoder was relaunched with a fresh slot 0.

    Regression for PR #109 review: after the decoder backend was disposed and
    re-created with the same ``N=1`` topology but a new ``qbruntime.Model``
    handle, the reset fast-path would silently retain the stale handle and
    dispatch would hit a disposed slot.
    """
    slot_0_v1 = object()
    model = _make_whisper_generation_stub(mxq_models=[slot_0_v1], max_batch_size=2)

    first_cache = model._get_cache("mobilint", batch_size=2, max_cache_len=1)
    assert first_cache.mxq_models == [slot_0_v1]

    slot_0_v2 = object()
    decoder = model.get_decoder()
    decoder.npu_backend.mxq_models = [slot_0_v2]
    decoder.npu_backend.mxq_model = slot_0_v2

    second_cache = model._get_cache("mobilint", batch_size=2, max_cache_len=1)
    assert second_cache is not first_cache
    assert second_cache.mxq_models == [slot_0_v2]


def test_whisper_get_cache_rebuilds_and_raises_on_multi_slot_relaunch() -> None:
    """A relaunch that grows the decoder to ``N > 1`` must go through rebuild.

    The reset fast-path would retain a Whisper cache built for ``N=1``,
    silently breaking the beam contract; instead the reuse guard rebuilds so
    the constructor-time ``n_slots>1`` guard fires with a clear
    ``NotImplementedError``.
    """
    slot_0_v1 = object()
    model = _make_whisper_generation_stub(mxq_models=[slot_0_v1], max_batch_size=2)

    model._get_cache("mobilint", batch_size=2, max_cache_len=1)

    new_slot_0, new_slot_1 = object(), object()
    decoder = model.get_decoder()
    decoder.npu_backend.mxq_models = [new_slot_0, new_slot_1]
    decoder.npu_backend.mxq_model = new_slot_0

    with pytest.raises(NotImplementedError, match="multi-slot dispatch"):
        model._get_cache("mobilint", batch_size=2, max_cache_len=1)
