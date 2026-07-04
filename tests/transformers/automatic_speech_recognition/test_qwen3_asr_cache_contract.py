"""Regression tests for Qwen3-ASR cache creation semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

pytest.importorskip(
    "qwen_asr",
    reason="Qwen3-ASR cache contract tests require the optional qwen-asr package.",
)

from mblt_model_zoo.hf_transformers.models.qwen3_asr.modeling_qwen3_asr import (  # noqa: E402
    MobilintQwen3ASRTextModel,
    MobilintQwen3ASRThinkerForConditionalGeneration,
)
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintBeamCache  # noqa: E402


class _InputsEmbedAwareMxqModel:
    """MXQ stub that makes logits depend on the supplied decoder embedding row."""

    def __init__(self) -> None:
        self.infer_calls = 0

    def get_model_output_shape(self) -> list[tuple[int, ...]]:
        # Dynamic token axis: matches the ``per-token logits`` shape this
        # stub returns from ``infer``, so ``_mxq_supports_all_logits`` picks
        # the ``supports_all`` branch of ``_beam_llm_forward`` and the
        # caller's ``prefill_chunk_size`` is preserved.
        return [(1, -1, 8)]

    def infer(self, inputs: list[object], output_buffer: object, cache_size: int) -> list[object]:
        """Return logits whose value identifies the forwarded embedding row."""
        del output_buffer, cache_size
        row_embeds = inputs[0]
        self.infer_calls += 1
        suffix_length = int(row_embeds.shape[2])
        row_value = float(row_embeds.mean())
        return [torch.full((suffix_length, 8), row_value, dtype=torch.float32).numpy()]


class _PositionAwareMxqModel:
    """MXQ stub that tags each returned row with its absolute cache position.

    Lets a test tell which sequence position ended up in the final tensor
    without having to reason about beam-cache internals: the logits value
    at row ``i`` is ``cache_size + i`` (broadcast across the vocab axis),
    so a selector's picked positions are directly readable from the output.
    """

    def __init__(self) -> None:
        self.infer_calls = 0

    def get_model_output_shape(self) -> list[tuple[int, ...]]:
        # Dynamic token axis; see ``_InputsEmbedAwareMxqModel``.
        return [(1, -1, 8)]

    def infer(self, inputs: list[object], output_buffer: object, cache_size: int) -> list[object]:
        del output_buffer
        row_embeds = inputs[0]
        self.infer_calls += 1
        suffix_length = int(row_embeds.shape[2])
        vocab = 8
        positions = torch.arange(cache_size, cache_size + suffix_length, dtype=torch.float32)
        return [positions.unsqueeze(-1).expand(suffix_length, vocab).contiguous().numpy()]


class _LastOnlyMxqModel:
    """MXQ stub compiled with a static token axis of 1 (last-token-only output).

    ``get_model_output_shape`` reports ``(1, 1, vocab)`` — no
    ``_MXQ_DYNAMIC_AXIS_SENTINEL`` in the token axis — so
    ``_mxq_supports_all_logits`` returns ``False`` and the beam path routes
    a non-default ``logits_to_keep`` request through the ``last_only_slow``
    branch. ``infer`` emits a single row per call regardless of chunk width,
    matching how a real last-only build behaves; the row value carries the
    absolute cache position of the last input token in the chunk so tests
    can assert which position ended up in the final tensor.
    """

    def __init__(self) -> None:
        self.infer_calls = 0
        self.vocab_size = 8

    def get_model_output_shape(self) -> list[tuple[int, ...]]:
        return [(1, 1, self.vocab_size)]

    def infer(self, inputs: list[object], output_buffer: object, cache_size: int) -> list[object]:
        del output_buffer
        row_embeds = inputs[0]
        self.infer_calls += 1
        chunk_len = int(row_embeds.shape[2])
        last_position = int(cache_size) + chunk_len - 1
        return [
            torch.full((1, self.vocab_size), float(last_position), dtype=torch.float32).numpy()
        ]


class _DummyCache:
    """Tiny cache stub exposing the sequence length API used by ``forward``."""

    def get_seq_length(self) -> int:
        """Return an empty cache length."""
        return 0


class _DummyQwen3ASRTextModel(MobilintQwen3ASRTextModel):
    """Minimal Qwen3-ASR text model that avoids NPU initialization."""

    def __init__(self, use_cache: bool) -> None:
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(use_cache=use_cache, vocab_size=8, hidden_size=4, pad_token_id=0)
        self.embed_tokens = torch.nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            self.config.pad_token_id,
        )
        self.cache_created = False
        self.forward_past_key_values = None

    def _get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, *args: object) -> object:
        """Record cache creation without allocating a real Mobilint cache."""
        del cache_implementation, batch_size, max_cache_len, args
        self.cache_created = True
        return _DummyCache()

    def llm_forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: object | None,
        cache_position: torch.Tensor,
        prefill_chunk_size: int | None = None,
        count_npu_time: bool = False,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        """Return deterministic logits while exposing forward-time cache state."""
        del cache_position, prefill_chunk_size, count_npu_time, attention_mask, logits_to_keep
        self.forward_past_key_values = past_key_values
        return torch.zeros(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            self.config.vocab_size,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )


class _DummyQwen3ASRBeamTextModel(MobilintQwen3ASRTextModel):
    """Minimal Qwen3-ASR text model for exercising beam decode directly."""

    def __init__(self, mxq_model: _InputsEmbedAwareMxqModel) -> None:
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(use_cache=True, vocab_size=8, hidden_size=2, pad_token_id=0)
        self.embed_tokens = torch.nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            self.config.pad_token_id,
        )
        self.npu_backend = SimpleNamespace(mxq_model=mxq_model)
        self.npu_time = None

    def resolve_prefill_chunk_size(self, prefill_chunk_size: int | None) -> int:
        """Use one chunk per test sequence unless explicitly overridden."""
        return prefill_chunk_size or 16


class _DummyQwen3ASRThinkerTextModel(torch.nn.Module):
    """Minimal nested text model for thinker output contract tests."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs = None

    def forward(self, **kwargs: object) -> BaseModelOutputWithPast:
        """Return deterministic hidden states while recording forwarded kwargs."""
        self.forward_kwargs = kwargs
        inputs_embeds = kwargs["inputs_embeds"]
        assert isinstance(inputs_embeds, torch.Tensor)
        return BaseModelOutputWithPast(last_hidden_state=inputs_embeds, past_key_values=None)


class _DummyQwen3ASRThinker(MobilintQwen3ASRThinkerForConditionalGeneration):
    """Minimal Qwen3-ASR thinker that avoids NPU initialization."""

    def __init__(self, return_dict: bool) -> None:
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(return_dict=return_dict)
        self.model = _DummyQwen3ASRThinkerTextModel()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Identity()
        self.rope_deltas = None

    def get_input_embeddings(self) -> torch.nn.Module:
        """Return dummy token embeddings."""
        return self.embed_tokens


@pytest.mark.parametrize("config_use_cache", [False, True])
def test_qwen3_asr_forward_does_not_force_cache_when_use_cache_is_false(config_use_cache: bool) -> None:
    """Respect explicit ``use_cache=False`` even for beam-expanded batches."""
    model = _DummyQwen3ASRTextModel(use_cache=config_use_cache)

    outputs = model(input_ids=torch.tensor([[1, 2], [1, 3]], dtype=torch.long), use_cache=False)

    assert model.cache_created is False
    assert model.forward_past_key_values is None
    assert outputs.past_key_values is None


def test_qwen3_asr_forward_creates_cache_for_multi_row_default_cache() -> None:
    """Keep the existing multi-row cache path when caching is not explicitly disabled."""
    model = _DummyQwen3ASRTextModel(use_cache=True)

    outputs = model(input_ids=torch.tensor([[1, 2], [1, 3]], dtype=torch.long))

    assert model.cache_created is True
    assert model.forward_past_key_values is outputs.past_key_values
    assert outputs.past_key_values is not None


def test_qwen3_asr_duplicate_logits_are_not_reused_across_audio_rows() -> None:
    """Identical token rows with different audio-conditioned embeddings should not share logits."""
    mxq_model = _InputsEmbedAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4], [4]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0]],
            [[7.0, 7.0]],
        ],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.tensor([0], dtype=torch.long),
    )

    assert mxq_model.infer_calls == 2
    assert logits.shape == (2, 1, 8)
    assert torch.equal(logits[0], torch.full((1, 8), 1.0))
    assert torch.equal(logits[1], torch.full((1, 8), 7.0))


def test_qwen3_asr_cached_suffix_keeps_audio_source_identity() -> None:
    """Keep source ids from the audio-conditioned prompt when later token suffixes match."""
    mxq_model = _InputsEmbedAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)

    model._beam_llm_forward(
        input_ids=torch.tensor([[4], [4]], dtype=torch.long),
        inputs_embeds=torch.tensor(
            [
                [[1.0, 1.0]],
                [[7.0, 7.0]],
            ],
            dtype=torch.float32,
        ),
        past_key_values=cache,
        cache_position=torch.tensor([0], dtype=torch.long),
    )

    logits = model._beam_llm_forward(
        input_ids=torch.tensor([[5], [5]], dtype=torch.long),
        inputs_embeds=torch.tensor(
            [
                [[3.0, 3.0]],
                [[3.0, 3.0]],
            ],
            dtype=torch.float32,
        ),
        past_key_values=cache,
        cache_position=torch.tensor([1], dtype=torch.long),
    )

    assert mxq_model.infer_calls == 4
    assert logits.shape == (2, 1, 8)


def test_qwen3_asr_duplicate_logits_reuse_single_source_beams() -> None:
    """Beam-expanded rows from the same audio-conditioned prompt may share logits."""
    mxq_model = _InputsEmbedAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4], [4]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[3.0, 3.0]],
            [[3.0, 3.0]],
        ],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.tensor([0], dtype=torch.long),
    )

    assert mxq_model.infer_calls == 1
    assert logits.shape == (2, 1, 8)
    assert torch.equal(logits[0], logits[1])


# ---------------------------------------------------------------------------
# Beam-path logits_to_keep contract
#
# The non-beam path (`llm_forward`) honors HF-style `logits_to_keep`, but the
# beam-cache-backed path (`_beam_llm_forward`) was silently returning the
# full input window regardless of the selector. These tests pin the same
# `(batch, kept, vocab)` shape contract that `TestLogitsShapeMatrix` enforces
# for the non-beam paths, and they verify that a tensor selector actually
# picks the requested absolute positions (via the position-aware stub) so a
# broken index-select would fail loudly instead of returning the right shape
# with wrong contents.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("logits_to_keep", "expected_kept"),
    [
        pytest.param(0, 3, id="int_0_keep_all"),
        pytest.param(1, 1, id="int_1_last_only"),
        pytest.param(2, 2, id="int_2_last_n"),
        pytest.param(torch.tensor([0, 2]), 2, id="tensor_pick_positions"),
        pytest.param(torch.tensor([-1]), 1, id="tensor_negative_last"),
        pytest.param(torch.tensor([1, 1]), 2, id="tensor_duplicate_positions"),
        pytest.param(torch.tensor([], dtype=torch.long), 0, id="tensor_empty"),
    ],
)
def test_qwen3_asr_beam_forward_honors_logits_to_keep_shape(
    logits_to_keep: object,
    expected_kept: int,
) -> None:
    """Beam path returns ``(batch, kept, vocab)`` matching HF selector semantics."""
    mxq_model = _InputsEmbedAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4, 5, 6], [4, 5, 6]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]],
        ],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(3, dtype=torch.long),
        logits_to_keep=logits_to_keep,
    )

    assert logits.shape == (2, expected_kept, 8)


def test_qwen3_asr_beam_forward_default_matches_pre_selector_behavior() -> None:
    """Default (``logits_to_keep=0``) preserves the ``(batch, input_ids.shape[1], vocab)``
    contract the beam path had before selector plumbing landed."""
    mxq_model = _InputsEmbedAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4], [4]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0]],
            [[7.0, 7.0]],
        ],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.tensor([0], dtype=torch.long),
    )

    assert logits.shape == (2, 1, 8)


def test_qwen3_asr_beam_forward_tensor_selector_picks_correct_positions() -> None:
    """A tensor selector must pick the requested absolute positions, not just
    a shape-compatible slice. Uses the position-aware stub so each row of the
    returned logits is tagged with its sequence index (``cache_size + i``)."""
    mxq_model = _PositionAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=1)
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
        dtype=torch.float32,
    )
    selector = torch.tensor([0, 3, 1])

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(4, dtype=torch.long),
        logits_to_keep=selector,
    )

    assert logits.shape == (1, 3, 8)
    # Row values are the absolute cache position (0..3) broadcast across vocab.
    assert torch.equal(logits[0, 0], torch.full((8,), 0.0))
    assert torch.equal(logits[0, 1], torch.full((8,), 3.0))
    assert torch.equal(logits[0, 2], torch.full((8,), 1.0))


def test_qwen3_asr_beam_forward_empty_selector_preserves_vocab_axis() -> None:
    """``torch.tensor([])`` returns ``(batch, 0, vocab)`` — the vocab axis
    survives so downstream stacking / lm_head passes see a consistent rank
    regardless of which path (beam vs. non-beam) produced the logits."""
    mxq_model = _InputsEmbedAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4, 5], [4, 5]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0]],
            [[7.0, 7.0], [8.0, 8.0]],
        ],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(2, dtype=torch.long),
        logits_to_keep=torch.tensor([], dtype=torch.long),
    )

    assert logits.shape == (2, 0, 8)
    assert logits.dtype == inputs_embeds.dtype


def test_qwen3_asr_beam_forward_out_of_range_selector_raises() -> None:
    """Out-of-range tensor indices must fail loudly, matching the non-beam
    contract (HF fancy-indexing raises IndexError on the same input)."""
    mxq_model = _InputsEmbedAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=1)
    input_ids = torch.tensor([[4, 5]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0]]],
        dtype=torch.float32,
    )

    with pytest.raises(IndexError):
        model._beam_llm_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            cache_position=torch.arange(2, dtype=torch.long),
            logits_to_keep=torch.tensor([99]),
        )


# ---------------------------------------------------------------------------
# Beam-path chunk-boundary contract
#
# When the suffix crosses a prefill-chunk boundary (``suffix_length >
# resolved_prefill_chunk_size``), the beam loop must accumulate every chunk's
# logits — not just the final chunk's — before the ``logits_to_keep`` selector
# runs. Without that, keep-all silently returns ``(batch, final_chunk_len,
# vocab)`` instead of ``(batch, input_ids.shape[1], vocab)`` and any tensor
# selector referencing a position outside the final chunk either raises
# ``IndexError`` or picks the wrong row. The shape-only tests above miss this
# because ``_DummyQwen3ASRBeamTextModel.resolve_prefill_chunk_size`` defaults
# to 16, well above the window sizes they use.
# ---------------------------------------------------------------------------


def test_qwen3_asr_beam_forward_multi_chunk_keeps_full_window() -> None:
    """Default keep-all across a chunk boundary must return the full window.

    With ``prefill_chunk_size=2`` and a 4-token window the suffix decodes over
    two chunks; the position-aware stub tags each row with its absolute cache
    position so the returned tensor reveals whether every chunk survived.
    """
    mxq_model = _PositionAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=1)
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(4, dtype=torch.long),
        prefill_chunk_size=2,
    )

    assert mxq_model.infer_calls == 2
    assert logits.shape == (1, 4, 8)
    for position in range(4):
        assert torch.equal(logits[0, position], torch.full((8,), float(position)))


def test_qwen3_asr_beam_forward_multi_chunk_selector_picks_early_position() -> None:
    """A tensor selector referencing a position inside the *first* chunk must
    still resolve to that position's logits after a multi-chunk suffix decode.
    Buggy code (final-chunk-only) either raises out-of-range or returns the
    wrong row when the selector reaches into an earlier chunk."""
    mxq_model = _PositionAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=1)
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
        dtype=torch.float32,
    )
    selector = torch.tensor([0, 3])

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(4, dtype=torch.long),
        prefill_chunk_size=2,
        logits_to_keep=selector,
    )

    assert logits.shape == (1, 2, 8)
    assert torch.equal(logits[0, 0], torch.full((8,), 0.0))
    assert torch.equal(logits[0, 1], torch.full((8,), 3.0))


def test_qwen3_asr_beam_forward_multi_chunk_empty_selector() -> None:
    """Empty-selector shape contract survives multi-chunk suffix decode."""
    mxq_model = _InputsEmbedAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=1)
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(4, dtype=torch.long),
        prefill_chunk_size=2,
        logits_to_keep=torch.tensor([], dtype=torch.long),
    )

    # Empty selector still requires the KV cache to advance through every
    # chunk — a fix that early-outs on empty selector would leave later
    # decode steps with an incomplete cache.
    assert mxq_model.infer_calls == 2
    assert logits.shape == (1, 0, 8)


def test_qwen3_asr_beam_forward_uneven_final_chunk_keeps_full_window() -> None:
    """Odd suffix (last chunk shorter than the rest) must still yield the
    full window. Guards against a fix that concatenates but assumes chunks
    are all the same size."""
    mxq_model = _PositionAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=1)
    input_ids = torch.tensor([[4, 5, 6]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(3, dtype=torch.long),
        prefill_chunk_size=2,
    )

    assert mxq_model.infer_calls == 2
    assert logits.shape == (1, 3, 8)
    for position in range(3):
        assert torch.equal(logits[0, position], torch.full((8,), float(position)))


def test_qwen3_asr_forward_routes_logits_to_keep_to_beam_path() -> None:
    """Verify ``forward`` plumbs ``logits_to_keep`` into the beam path — the
    review-flagged bug was that the wrapper silently dropped the selector
    when routing to ``_beam_llm_forward``."""
    mxq_model = _InputsEmbedAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4, 5, 6], [4, 5, 6]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]],
        ],
        dtype=torch.float32,
    )

    outputs = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        logits_to_keep=1,
    )

    assert outputs.last_hidden_state.shape == (2, 1, 8)


def test_qwen3_asr_thinker_forward_supports_return_dict_false() -> None:
    """Preserve upstream tuple output contract when ``return_dict=False`` is explicit."""
    model = _DummyQwen3ASRThinker(return_dict=True)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    outputs = model(input_ids=input_ids, return_dict=False)

    assert isinstance(outputs, tuple)
    assert torch.equal(outputs[0], model.get_input_embeddings()(input_ids))
    assert "return_dict" not in model.model.forward_kwargs


def test_qwen3_asr_thinker_forward_uses_config_return_dict_default() -> None:
    """Use the thinker config default when ``return_dict`` is omitted."""
    model = _DummyQwen3ASRThinker(return_dict=False)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    outputs = model(input_ids=input_ids)

    assert isinstance(outputs, tuple)
    assert torch.equal(outputs[0], model.get_input_embeddings()(input_ids))


# ---------------------------------------------------------------------------
# Beam-path selector contract under active-cache prefix reuse
#
# When two beams share an audio source (identical ``inputs_embeds`` rows) but
# their ``input_ids`` share only a proper prefix, the first beam commits an
# active cache the second beam partially reuses. Concretely: beam 0 forwards
# ``[A, B, X]`` (active cache becomes ``[A, B, X]``), and beam 1's target
# ``[A, B, Y]`` yields ``prefix_length=2 > previous_length=0``, so beam 1's
# ``local_start_index=2`` and only 1 suffix row is decoded. The resulting
# ``row_logits`` for beam 1 spans window positions ``[2, 3)`` only; window
# positions ``[0, 2)`` were not decoded this call. The selector must either
# limit itself to the decoded suffix or raise ``IndexError`` — silently
# returning a shortened tensor or picking the wrong absolute row is what
# these tests guard against.
# ---------------------------------------------------------------------------


def test_beam_forward_prefix_reuse_keep_all_raises() -> None:
    """``logits_to_keep=0`` (keep-all) must fail when the beam-cache prefix
    covers part of the input window — the pre-fix code silently returned a
    ``(batch, suffix_length, vocab)`` tensor shorter than the window."""
    mxq_model = _PositionAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4, 5, 6], [4, 5, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(IndexError, match="beam cache offset"):
        model._beam_llm_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            cache_position=torch.arange(3, dtype=torch.long),
            logits_to_keep=0,
        )


def test_beam_forward_prefix_reuse_last_one_ok() -> None:
    """``logits_to_keep=1`` (HF ``.generate()`` default) always succeeds
    because the last window position is inside every beam's decoded suffix.
    The position-aware stub tags each row with its absolute cache position,
    so the returned value equals the last input window position."""
    mxq_model = _PositionAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4, 5, 6], [4, 5, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(3, dtype=torch.long),
        logits_to_keep=1,
    )

    assert logits.shape == (2, 1, 8)
    assert torch.equal(logits[0, 0], torch.full((8,), 2.0))
    assert torch.equal(logits[1, 0], torch.full((8,), 2.0))


def test_beam_forward_prefix_reuse_last_n_within_suffix_ok() -> None:
    """``logits_to_keep=n`` where ``n`` equals the aligned suffix length
    returns ``(batch, n, vocab)`` with row values matching the tail window
    positions. Setup: shared prefix of length 1 → ``beam_offset=1``, so the
    aligned suffix has 2 rows and ``n=2`` is fully inside it."""
    mxq_model = _PositionAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4, 5, 6], [4, 6, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(3, dtype=torch.long),
        logits_to_keep=2,
    )

    assert logits.shape == (2, 2, 8)
    for beam in range(2):
        assert torch.equal(logits[beam, 0], torch.full((8,), 1.0))
        assert torch.equal(logits[beam, 1], torch.full((8,), 2.0))


def test_beam_forward_prefix_reuse_tensor_selector_in_suffix_ok() -> None:
    """A tensor selector referencing only positions inside every beam's
    decoded suffix (``>= beam_offset``) returns the absolute-position rows,
    not the ``[0]`` of the trimmed tensor. Duplicate positions are preserved."""
    mxq_model = _PositionAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4, 5, 6], [4, 6, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    selector = torch.tensor([2, 1, 2])

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(3, dtype=torch.long),
        logits_to_keep=selector,
    )

    assert logits.shape == (2, 3, 8)
    for beam in range(2):
        assert torch.equal(logits[beam, 0], torch.full((8,), 2.0))
        assert torch.equal(logits[beam, 1], torch.full((8,), 1.0))
        assert torch.equal(logits[beam, 2], torch.full((8,), 2.0))


def test_beam_forward_prefix_reuse_tensor_selector_below_offset_raises() -> None:
    """A tensor selector referencing a position covered only by the active
    beam cache (``< beam_offset``) raises ``IndexError`` — the pre-fix code
    silently returned the wrong absolute row via ``index_select`` on the
    trimmed tensor."""
    mxq_model = _PositionAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    input_ids = torch.tensor([[4, 5, 6], [4, 5, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(IndexError, match="beam cache offset"):
        model._beam_llm_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            cache_position=torch.arange(3, dtype=torch.long),
            logits_to_keep=torch.tensor([0]),
        )


def test_beam_forward_prefix_reuse_heterogeneous_beams_align() -> None:
    """Two beams with different ``local_start_index`` must align to the
    largest offset so ``torch.cat`` succeeds; pre-fix code left the two
    ``row_logits`` at mismatched ``shape[1]`` and either raised a cryptic
    size error or (with a lucky shape) picked wrong rows. Under
    ``logits_to_keep=1`` the aligned last-position rows carry each beam's
    absolute last-window position."""
    mxq_model = _PositionAwareMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=2)
    # Beam 0 has no shared prefix (it forwards first, so its own history
    # is empty and prefix_length=0 → local_start_index=0). Beam 1 shares
    # a 2-token prefix with beam 0's committed active tokens →
    # local_start_index=2. Alignment must front-trim beam 0 to shape (1,1,vocab).
    input_ids = torch.tensor([[4, 5, 6], [4, 5, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(3, dtype=torch.long),
        logits_to_keep=1,
    )

    # Aligned suffix length = window_length - max(local_start_index) = 3 - 2 = 1.
    assert logits.shape == (2, 1, 8)
    # Last window position is index 2 for both beams; position-aware stub
    # tags each row with its absolute cache position.
    assert torch.equal(logits[0, 0], torch.full((8,), 2.0))
    assert torch.equal(logits[1, 0], torch.full((8,), 2.0))


# ---------------------------------------------------------------------------
# Beam-path last-only MXQ contract
#
# A last-only MXQ build compiles the LM head with a static token axis of 1,
# so every ``mxq_model.infer`` call returns one row regardless of chunk
# width. Before the ``_mxq_supports_all_logits`` dispatch landed in the beam
# path, a multi-chunk suffix on such a backend produced ``num_chunks`` rows
# instead of ``suffix_length`` rows: ``logits_to_keep=0`` returned a
# truncated tensor and tensor / last-N selectors either raised
# ``IndexError`` or picked from the wrong shortened axis. These tests pin
# the fix in place:
#
# * ``logits_to_keep=1`` hits the last-token fast path and preserves the
#   caller's ``prefill_chunk_size`` — only ``ceil(suffix_length /
#   prefill_chunk_size)`` infer calls, and the returned row is the last
#   window position on both static and dynamic backends.
# * ``logits_to_keep=0`` (and any non-default selector) hits the size-1
#   fallback so each infer emits one row per suffix position, matching
#   the shape contract the outer alignment / selector logic already
#   expects. Slow-path warning fires once per model instance.
# ---------------------------------------------------------------------------


def test_beam_forward_last_only_backend_logits_to_keep_1_uses_fast_path() -> None:
    """``logits_to_keep=1`` on a last-only MXQ preserves the caller's
    ``prefill_chunk_size`` (no size-1 override) and returns the last-token
    logit. Pre-fix code either raised ``IndexError`` on the outer
    ``index_select`` or picked from the wrong shortened axis when the
    suffix crossed a chunk boundary."""
    mxq_model = _LastOnlyMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=1)
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
        dtype=torch.float32,
    )

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(4, dtype=torch.long),
        prefill_chunk_size=2,
        logits_to_keep=1,
    )

    # Fast path: 2 chunks of the caller's ``prefill_chunk_size=2``.
    assert mxq_model.infer_calls == 2
    assert logits.shape == (1, 1, 8)
    # Last window position is index 3 (``cache_size + chunk_len - 1`` for
    # the final chunk).
    assert torch.equal(logits[0, 0], torch.full((8,), 3.0))


def test_beam_forward_last_only_backend_keep_all_uses_size_1_fallback() -> None:
    """``logits_to_keep=0`` on a last-only MXQ forces ``effective_chunk_size
    = 1`` so each infer captures one row per suffix position. The returned
    tensor must span the full input window with each row tagged by its
    absolute position — pre-fix code returned ``(1, num_chunks, vocab)``
    with silently truncated content."""
    mxq_model = _LastOnlyMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=1)
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
        dtype=torch.float32,
    )

    with pytest.warns(UserWarning, match="last-only MXQ"):
        logits = model._beam_llm_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            cache_position=torch.arange(4, dtype=torch.long),
            prefill_chunk_size=2,
            logits_to_keep=0,
        )

    # Size-1 override: one infer per suffix position.
    assert mxq_model.infer_calls == 4
    assert logits.shape == (1, 4, 8)
    for position in range(4):
        assert torch.equal(logits[0, position], torch.full((8,), float(position)))


def test_beam_forward_last_only_backend_tensor_selector_picks_absolute_positions() -> None:
    """A tensor selector on a last-only MXQ still resolves to the absolute
    positions requested — the size-1 fallback gives the outer selector
    ``(1, suffix_length, vocab)`` rows to index against."""
    mxq_model = _LastOnlyMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=1)
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
        dtype=torch.float32,
    )
    selector = torch.tensor([0, 3, 1])

    logits = model._beam_llm_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=torch.arange(4, dtype=torch.long),
        prefill_chunk_size=2,
        logits_to_keep=selector,
    )

    assert mxq_model.infer_calls == 4
    assert logits.shape == (1, 3, 8)
    assert torch.equal(logits[0, 0], torch.full((8,), 0.0))
    assert torch.equal(logits[0, 1], torch.full((8,), 3.0))
    assert torch.equal(logits[0, 2], torch.full((8,), 1.0))


def test_beam_forward_last_only_backend_last_only_selector_does_not_warn() -> None:
    """The slow-path warning is scoped to the ``last_only_slow`` branch: a
    ``logits_to_keep=1`` request (fast path) must not emit it."""
    mxq_model = _LastOnlyMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    cache = MobilintBeamCache(mxq_model, batch_size=1)
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    inputs_embeds = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]],
        dtype=torch.float32,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model._beam_llm_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            cache_position=torch.arange(4, dtype=torch.long),
            prefill_chunk_size=2,
            logits_to_keep=1,
        )


def test_beam_forward_last_only_backend_warns_only_once_per_instance() -> None:
    """The slow-path warning fires at most once per model instance, matching
    ``_warn_last_only_slow_path_once``'s contract."""
    mxq_model = _LastOnlyMxqModel()
    model = _DummyQwen3ASRBeamTextModel(mxq_model)
    input_ids = torch.tensor([[4, 5]], dtype=torch.long)
    inputs_embeds = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]], dtype=torch.float32)

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        # First call fires the warning.
        model._beam_llm_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=MobilintBeamCache(mxq_model, batch_size=1),
            cache_position=torch.arange(2, dtype=torch.long),
            prefill_chunk_size=2,
            logits_to_keep=0,
        )
        # Second call — same instance — must not re-emit it.
        model._beam_llm_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=MobilintBeamCache(mxq_model, batch_size=1),
            cache_position=torch.arange(2, dtype=torch.long),
            prefill_chunk_size=2,
            logits_to_keep=0,
        )

    last_only_warnings = [w for w in caught if "last-only MXQ" in str(w.message)]
    assert len(last_only_warnings) == 1
