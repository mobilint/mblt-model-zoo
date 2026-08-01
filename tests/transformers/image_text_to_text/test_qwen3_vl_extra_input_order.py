"""Regression tests for the Qwen3-VL text MXQ input-count detection + pack layout.

Supported compiled layouts (each dispatch honors its own layout — the
3-input orders below are intentionally different):

* Non-batch (``max_batch_size == 1``): 2 or 3 inputs.
  * 2-input ``[inputs (1,-1,H), deepstack (num_layers,-1,H)]`` — legacy/static
    (2B/4B W8): MRoPE baked into the compiled model, no rope tensor is fed.
  * 3-input ``[inputs (1,-1,H), deepstack (num_layers,-1,H), rope (1,-1,peSize)]``
    — dynamic (8B W8 shipped on HF Hub): rope threaded through ``_do_infer``
    after deepstack.
* Batch (``max_batch_size > 1``, current Batch16 W8): 3 inputs
  ``[inputs (1,-1,H), rope (1,-1,peSize), deepstack (num_layers,-1,H)]``.
  Legacy 2-input batch MXQ is no longer supported (hard-failed).

The input-count detection used to key off ``len(get_input_buffer_info())``,
which collapses to ``1`` on batch builds (all inputs fuse into a single
buffer). We use the variant handle's ``get_model_input_shape()`` instead —
it reports one entry per tensor input on every layout. The tests below cover
count detection for all three shipped shapes and end-to-end pack-order
assertions for both 3-input dispatches (non-batch ``_do_infer`` and batched
``_llm_forward_batch_deepstack``).
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
        # Non-batch W8 (2B/4B): [inputs, deepstack]
        ([(1, -1, 4096), (3, -1, 4096)], 2),
        # Non-batch W8 (8B on HF Hub): [inputs, deepstack, rope]
        ([(1, -1, 4096), (3, -1, 4096), (1, -1, 256)], 3),
        # Batch16 W8: [inputs, rope, deepstack] — different rope position
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
# End-to-end pack-order checks. The two 3-input dispatches emit different
# extras orders on purpose (each matches the actual compiled MXQ):
#   * non-batch ``_do_infer``: ``[inputs, deepstack, rope]``
#   * batched ``_llm_forward_batch_deepstack``: ``[inputs, rope, deepstack]``
# ---------------------------------------------------------------------------


class _RecordingSingleBatchMxq:
    """MXQ stub that captures every single-batch ``infer`` call for assertions."""

    def __init__(self, vocab_size: int = 5):
        self.vocab_size = vocab_size
        self.calls: list[dict] = []

    def infer(self, inputs, _extra, cache_size):
        self.calls.append(
            {
                "shapes": [tuple(np.asarray(x).shape) for x in inputs],
                "cache_size": int(cache_size),
            }
        )
        seq = int(np.asarray(inputs[0]).shape[-2])
        return [np.zeros((1, seq, self.vocab_size), dtype=np.float32)]


def _make_single_batch_model(
    *,
    uses_rope_input: bool,
    hidden_size: int = 4,
    num_deepstack_layers: int = 3,
) -> MobilintQwen3VLTextModel:
    """Build a ``MobilintQwen3VLTextModel`` stub wired for the non-batch path."""
    mxq = _RecordingSingleBatchMxq(vocab_size=5)
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


def test_single_batch_pack_2input_omits_rope() -> None:
    """Legacy non-batch 2-input MXQ: ``_do_infer`` emits ``[inputs, deepstack]``."""
    hidden_size = 4
    num_layers = 3
    seq_len = 2

    model = _make_single_batch_model(
        uses_rope_input=False,
        hidden_size=hidden_size,
        num_deepstack_layers=num_layers,
    )
    inputs_embeds = torch.zeros((1, seq_len, hidden_size), dtype=torch.float32)

    model.llm_forward(
        inputs_embeds=inputs_embeds,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
        past_key_values=None,
        cache_position=torch.arange(seq_len),
        npu_prefill_chunk_size=seq_len,
        count_npu_time=False,
        logits_to_keep=1,
        position_embeddings=None,
    )

    mxq: _RecordingSingleBatchMxq = model._recording_mxq  # type: ignore[assignment]
    assert len(mxq.calls) == 1
    shapes = mxq.calls[0]["shapes"]
    assert len(shapes) == 2, f"expected 2 inputs on legacy non-batch MXQ, got {shapes}"
    # position 0: inputs_embeds — (1, seq_len, hidden)
    assert shapes[0] == (1, seq_len, hidden_size)
    # position 1: deepstack — (num_layers, seq_len, hidden)
    assert shapes[1] == (num_layers, seq_len, hidden_size)


def test_single_batch_pack_3input_emits_deepstack_then_rope() -> None:
    """Non-batch 3-input MXQ (8B on HF Hub): order is ``[inputs, deepstack, rope]``.

    Deliberately different from the batched build's ``[inputs, rope,
    deepstack]``: the two compiled signatures ship with different tensor
    orders and each dispatch honors its own layout.
    """
    hidden_size = 4
    peSize = 8
    num_layers = 3
    seq_len = 2

    model = _make_single_batch_model(
        uses_rope_input=True,
        hidden_size=hidden_size,
        num_deepstack_layers=num_layers,
    )
    inputs_embeds = torch.zeros((1, seq_len, hidden_size), dtype=torch.float32)
    position_embeddings = np.zeros((1, seq_len, peSize), dtype=np.float32)

    model.llm_forward(
        inputs_embeds=inputs_embeds,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
        past_key_values=None,
        cache_position=torch.arange(seq_len),
        npu_prefill_chunk_size=seq_len,
        count_npu_time=False,
        logits_to_keep=1,
        position_embeddings=position_embeddings,
    )

    mxq: _RecordingSingleBatchMxq = model._recording_mxq  # type: ignore[assignment]
    assert len(mxq.calls) == 1
    shapes = mxq.calls[0]["shapes"]
    assert len(shapes) == 3, f"expected 3 inputs on non-batch 3-input MXQ, got {shapes}"
    # position 0: inputs_embeds — (1, seq_len, hidden)
    assert shapes[0] == (1, seq_len, hidden_size)
    # position 1: deepstack — (num_layers, seq_len, hidden)
    assert shapes[1] == (num_layers, seq_len, hidden_size)
    # position 2: rope — (1, seq_len, peSize)
    assert shapes[2] == (1, seq_len, peSize)


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
