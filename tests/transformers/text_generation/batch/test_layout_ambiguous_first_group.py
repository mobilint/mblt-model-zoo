"""Regression test for the multi-slot merge layout latch bug.

Prior to the :class:`MultiSlotDispatcher` refactor, the batched-infer merge
inferred its output layout (per-item last row vs per-token flat) from the
FIRST non-empty group in each dispatch. If that group was all ``seq_len == 1``
rows the two candidate row counts collapsed (``n_items == n_tokens``) and the
merge defaulted to layout A silently — later groups with longer sequences then
had their per-token logits truncated to the first row.

The fix pins the output layout once from the compiled MXQ shape probe on the
backend, so the merge no longer guesses per dispatch. This test constructs a
K=1 multi-slot backend where slot 0 receives a ``seq_len == 1`` row (the
ambiguous group) and slot 1 receives a longer prompt whose per-token logits
must survive the merge intact.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache
from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin


class _DynamicAxisMxq:
    """Dynamic-token-axis MXQ stub that emits per-token flat logits.

    ``get_model_output_shape()`` reports ``-1`` on the token axis so the
    backend's shape probe pins ``output_layout='n_tokens'``. Every ``.infer``
    call returns ``(total_tokens, vocab)`` — the layout that the pre-refactor
    merge silently truncated when the first group was decode-shaped.
    """

    def __init__(self, vocab_size: int = 5, max_width: int = 8, token_offset: float = 0.0) -> None:
        self.vocab_size = vocab_size
        self.max_width = max_width
        self.token_offset = float(token_offset)
        self.calls: list[dict] = []

    def get_input_buffer_info(self):
        class _Info:
            def __init__(self, max_width: int) -> None:
                self.max_width = max_width
                self.max_cache_size = 128

        return [_Info(self.max_width)]

    def get_model_output_shape(self):
        # Dynamic token axis; the shared helper interprets this as per-token.
        return [(1, -1, self.vocab_size)]

    def infer(self, inputs, _extra, cache_size, batch_params=None):
        chunk = np.asarray(inputs[0])
        assert batch_params is not None, "layout regression test targets the batched path"
        total_tokens = sum(int(p.sequence_length) for p in batch_params)
        # Encode ``(row_idx_within_group, token_idx_within_row)`` into every
        # vocab entry so the assertion below can spot layout truncation.
        rows: list[np.ndarray] = []
        for local_row, param in enumerate(batch_params):
            seq_len = int(param.sequence_length)
            for token_idx in range(seq_len):
                value = self.token_offset + 100.0 * local_row + token_idx
                rows.append(np.full((self.vocab_size,), value, dtype=np.float32))
        flat = np.stack(rows, axis=0)
        assert flat.shape == (total_tokens, self.vocab_size)
        self.calls.append(
            {
                "shape": tuple(chunk.shape),
                "batch": [(p.cache_id, int(p.sequence_length), int(p.cache_size)) for p in batch_params],
            }
        )
        return [flat]


class _MultiSlotDynamicBackend:
    """Fake multi-slot backend that surfaces ``output_layout='n_tokens'`` via probe."""

    def __init__(self, mxq_models: List[_DynamicAxisMxq]) -> None:
        self.mxq_models = list(mxq_models)
        self.mxq_model = self.mxq_models[0]
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
            from mblt_model_zoo.hf_transformers.utils.multi_slot_dispatch import MultiSlotDispatcher

            self._dispatcher = MultiSlotDispatcher(self)
        return self._dispatcher


def _make_model(mxq_models: List[_DynamicAxisMxq]) -> MobilintModelMixin:
    model = MobilintModelMixin.__new__(MobilintModelMixin)
    model.npu_backend = _MultiSlotDynamicBackend(mxq_models)
    model.config = type(
        "Config",
        (),
        {
            "npu_prefill_chunk_size": None,
            "max_batch_size": len(mxq_models),
            "vocab_size": mxq_models[0].vocab_size,
        },
    )()
    model.npu_time = None
    return model


def test_multi_slot_merge_preserves_longer_prompt_when_first_group_is_size_one() -> None:
    """Row 0 (seq_len==1) must not force layout A and truncate row 1's per-token logits.

    K=1 multi-slot backend: row 0 -> slot 0 (seq_len==1, ambiguous group), row 1 ->
    slot 1 (seq_len==3, unambiguous group). Pre-refactor the merge locked onto
    layout A from the first group and dropped the last two rows of row 1's
    per-token logits.
    """
    m0 = _DynamicAxisMxq(vocab_size=4, max_width=8, token_offset=0.0)
    m1 = _DynamicAxisMxq(vocab_size=4, max_width=8, token_offset=1000.0)
    model = _make_model([m0, m1])
    cache = MobilintCache([m0, m1], per_model_batch=1)

    # Row 0 has one active token (decode-shaped); row 1 has three active tokens
    # (prefill). Left-padded row 0 uses attention_mask to hide the pad.
    seq_len = 3
    hidden = 4
    inputs_embeds = torch.randn(2, seq_len, hidden, dtype=torch.float32)
    attention_mask = torch.tensor(
        [
            [0, 0, 1],  # row 0: only the last position is active
            [1, 1, 1],  # row 1: all three positions active
        ],
        dtype=torch.long,
    )
    cache_position = torch.arange(seq_len)

    # ``logits_to_keep=0`` (keep-all) drives the dynamic-axis batched path
    # (Path 2) so we see the per-token merge in action.
    logits = model.llm_forward(
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        cache_position=cache_position,
        attention_mask=attention_mask,
        logits_to_keep=0,
    )

    # Path 2 stacks per-item kept-position tensors and right-pads shorter rows
    # with -inf so the output has shape ``(batch, max_kept, vocab)``. Every
    # item keeps its full active-token window here (row 0 keeps 1 position;
    # row 1 keeps 3), so row 0 gets padded up to 3.
    assert logits.shape == (2, 3, 4)

    # Row 0 (from m0): first kept position is token 0 of its only active row
    # -> value 0.0 (m0 token_offset=0, local_row=0, token_idx=0). Remaining
    # slots are the -inf padding.
    assert logits[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert torch.all(logits[0, 1:] == float("-inf"))

    # Row 1 (from m1): three kept positions carrying token_offset=1000 + token_idx.
    # This is exactly the sequence the pre-refactor code truncated: if the merge
    # picked layout A after the first group, only row 1's first token would land
    # here and the last two would be lost / undefined.
    assert logits[1, 0].tolist() == [1000.0, 1000.0, 1000.0, 1000.0]
    assert logits[1, 1].tolist() == [1001.0, 1001.0, 1001.0, 1001.0]
    assert logits[1, 2].tolist() == [1002.0, 1002.0, 1002.0, 1002.0]

    # Each slot dispatched exactly one batched infer.
    assert len(m0.calls) == 1
    assert len(m1.calls) == 1
    # Slot 0 saw a size-1 group (the ambiguous one); slot 1 saw a size-3 group.
    assert m0.calls[0]["batch"] == [(0, 1, 0)]
    assert m1.calls[0]["batch"] == [(0, 3, 0)]
