"""Regression tests for the Qwen3-VL text MXQ input-count detection + pack layout.

Two compiled layouts are supported and they are disjoint by ``max_batch_size``:

* Non-batch W8 (2B/4B/8B): 2 inputs
  ``[inputs_embeds (1,-1,H), deepstack (num_layers,-1,H)]``. No external
  rope tensor.
* Batch16 W8: 3 inputs ``[inputs_embeds (1,-1,H), rope (1,-1,peSize),
  deepstack (num_layers,-1,H)]``. Requires the rope table produced by
  :class:`MobilintQwen3VLRotaryEmbedding`.

The input-count detection used to key off ``len(get_input_buffer_info())``,
which collapses to ``1`` on batch builds (all inputs fuse into a single
buffer). This regression uses the variant handle's
``get_model_input_shape()`` instead — that reports one entry per tensor
input on both batch and non-batch layouts — and the batched forward now
hardcodes the ``[rope, deepstack]`` extras order that the shipped 3-input
Batch16 MXQ expects.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pytest
import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLTextModel,
)


class _FakeVariantHandle:
    def __init__(self, shapes: Sequence[tuple[int, ...]]):
        self._shapes = list(shapes)

    def get_model_input_shape(self) -> list[tuple[int, ...]]:
        return list(self._shapes)


class _FakeMxq:
    def __init__(self, shapes: Sequence[tuple[int, ...]]):
        self._handle = _FakeVariantHandle(shapes)

    def get_model_variant_handle(self, idx: int) -> _FakeVariantHandle:
        assert idx == 0
        return self._handle


class _BareTextModel(MobilintQwen3VLTextModel):
    """Bypass NPU init; expose ``_get_num_mxq_inputs`` on a hand-set fake."""

    def __init__(self, mxq: _FakeMxq):
        torch.nn.Module.__init__(self)
        self._fake_mxq = mxq

    def get_mxq_model(self) -> _FakeMxq:  # type: ignore[override]
        return self._fake_mxq


@pytest.mark.parametrize(
    ("shapes", "expected_count"),
    [
        # Non-batch W8: [inputs, deepstack]
        ([(1, -1, 4096), (3, -1, 4096)], 2),
        # Batch16 W8: [inputs, rope, deepstack]
        ([(1, -1, 4096), (1, -1, 256), (3, -1, 4096)], 3),
    ],
)
def test_get_num_mxq_inputs_reads_variant_handle(
    shapes: list[tuple[int, ...]], expected_count: int
) -> None:
    """Counts must come from the variant handle, not ``get_input_buffer_info()``.

    ``get_input_buffer_info()`` returns a single fused entry on batch
    builds, so a naive ``len(get_input_buffer_info())`` reads 1 regardless
    of the actual tensor count. The variant handle reports the true count.
    """
    model = _BareTextModel(_FakeMxq(shapes))
    assert model._get_num_mxq_inputs() == expected_count


# ---------------------------------------------------------------------------
# End-to-end pack-order check: verify the batched extras list emitted by
# ``_llm_forward_batch_deepstack`` is ``[rope, deepstack]`` — the order the
# shipped 3-input Batch16 MXQ expects.
# ---------------------------------------------------------------------------


class _RecordingBatchMxq:
    """MXQ stub that captures every batched ``infer`` call for assertions."""

    def __init__(self, vocab_size: int = 5, max_width: int = 4):
        self.vocab_size = vocab_size
        self.max_width = max_width
        self.calls: list[dict] = []

    def get_input_buffer_info(self):
        # Batch builds always report a single fused buffer regardless of
        # the true tensor count (the whole reason the count comes from the
        # variant handle instead).
        return [SimpleNamespace(max_width=self.max_width, max_cache_size=0)]

    def get_model_output_shape(self):
        return [(1, -1, self.vocab_size)]

    def infer(self, inputs, _extra, cache_size, batch_params=None):
        self.calls.append(
            {
                "shapes": [tuple(np.asarray(x).shape) for x in inputs],
                "cache_size": int(cache_size),
                "batch": None
                if batch_params is None
                else [(p.cache_id, p.sequence_length, p.cache_size) for p in batch_params],
            }
        )
        # Return per-token flat logits so the shared LLM core's Path 1 layout
        # detector accepts the result.
        total_tokens = 0
        for p in batch_params or []:
            total_tokens += int(p.sequence_length)
        if total_tokens == 0:
            total_tokens = int(np.asarray(inputs[0]).shape[-2])
        payload = np.zeros((total_tokens, self.vocab_size), dtype=np.float32)
        return [payload]


def _make_batched_model(
    *,
    uses_rope_input: bool,
    hidden_size: int = 4,
    num_deepstack_layers: int = 3,
) -> MobilintQwen3VLTextModel:
    """Build a ``MobilintQwen3VLTextModel`` stub wired for the batched path."""
    mxq = _RecordingBatchMxq(vocab_size=5, max_width=4)
    model = MobilintQwen3VLTextModel.__new__(MobilintQwen3VLTextModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vocab_size=5,
        hidden_size=hidden_size,
        pad_token_id=0,
        npu_prefill_chunk_size=None,
        max_batch_size=1,
        use_cache=False,
    )
    model.npu_backend = SimpleNamespace(mxq_model=mxq)
    model.num_deepstack_layers = num_deepstack_layers
    model.npu_time = None
    model._uses_rope_input = uses_rope_input
    model.rotary_emb = None
    model._recording_mxq = mxq
    return model


def test_batched_pack_emits_rope_then_deepstack() -> None:
    """The shipped 3-input Batch16 layout is ``[inputs, rope, deepstack]``."""
    hidden_size = 4
    peSize = 8
    num_layers = 3
    seq_len = 2

    model = _make_batched_model(
        uses_rope_input=True,
        hidden_size=hidden_size,
        num_deepstack_layers=num_layers,
    )
    inputs_embeds = torch.zeros((1, seq_len, hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)
    position_embeddings = np.zeros((1, seq_len, peSize), dtype=np.float32)

    model._llm_forward_batch_deepstack(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
        past_key_values=None,
        cache_position=torch.arange(seq_len),
        npu_prefill_chunk_size=seq_len,
        count_npu_time=False,
        logits_to_keep=1,
        position_embeddings=position_embeddings,
    )

    mxq: _RecordingBatchMxq = model._recording_mxq  # type: ignore[assignment]
    assert len(mxq.calls) == 1
    shapes = mxq.calls[0]["shapes"]
    # position 0: inputs_embeds — (1, packed_tokens, hidden)
    assert shapes[0] == (1, seq_len, hidden_size)
    # position 1: rope — (1, packed_tokens, peSize)
    assert shapes[1] == (1, seq_len, peSize)
    # position 2: deepstack — (num_layers, packed_tokens, hidden)
    assert shapes[2] == (num_layers, seq_len, hidden_size)


def test_batched_path_rejects_2_input_mxq() -> None:
    """Legacy 2-input batch MXQ (old HF Batch16 W4V8) is no longer supported."""
    hidden_size = 4
    seq_len = 2

    model = _make_batched_model(uses_rope_input=False, hidden_size=hidden_size)
    inputs_embeds = torch.zeros((1, seq_len, hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)

    with pytest.raises(ValueError, match="requires a 3-input"):
        model._llm_forward_batch_deepstack(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            deepstack_visual_embeds=None,
            visual_pos_masks=None,
            past_key_values=None,
            cache_position=torch.arange(seq_len),
            npu_prefill_chunk_size=seq_len,
            count_npu_time=False,
            logits_to_keep=1,
            position_embeddings=None,
        )
