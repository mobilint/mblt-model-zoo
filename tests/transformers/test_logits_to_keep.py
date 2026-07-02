"""Regression tests for the shared ``logits_to_keep`` path in ``MobilintModelMixin``.

Covers three code branches introduced by the HF-style ``logits_to_keep``
support in the shared LLM inference core:

1. Fast path (``logits_to_keep == 1``): the original last-token behavior.
2. Dynamic-axis MXQ path (``mxq_model.get_model_output_shape()[0][-2] == -1``):
   the compiled model emits per-position logits, so all-chunk outputs are
   concatenated then sliced.
3. Last-only MXQ fallback: prefill non-kept prefix through normal chunks and
   run a size-1 infer at each kept position.

Both the single-item and batched paths are exercised via fake MXQ backends.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest
import torch

from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin
from tests.transformers._fake_mxq import (
    DynamicAxisMxq,
    StaticLastOnlyMxq,
    make_model,
)


# ---------------------------------------------------------------------------
# _normalize_logits_to_keep
# ---------------------------------------------------------------------------


class TestNormalizeLogitsToKeep:
    def test_int_zero_keeps_every_position(self) -> None:
        assert MobilintModelMixin._normalize_logits_to_keep(0, 5) == [0, 1, 2, 3, 4]

    def test_int_one_keeps_last_position(self) -> None:
        assert MobilintModelMixin._normalize_logits_to_keep(1, 5) == [4]

    def test_int_n_keeps_last_n_positions(self) -> None:
        assert MobilintModelMixin._normalize_logits_to_keep(3, 5) == [2, 3, 4]

    def test_int_oversized_keeps_all_positions(self) -> None:
        assert MobilintModelMixin._normalize_logits_to_keep(9, 4) == [0, 1, 2, 3]

    def test_int_equal_seq_len_keeps_all_positions(self) -> None:
        assert MobilintModelMixin._normalize_logits_to_keep(5, 5) == [0, 1, 2, 3, 4]

    def test_int_negative_falls_back_to_all_positions(self) -> None:
        assert MobilintModelMixin._normalize_logits_to_keep(-1, 3) == [0, 1, 2]

    def test_tensor_indices_are_sorted_and_deduped(self) -> None:
        indices = torch.tensor([3, 0, 3, 2])
        assert MobilintModelMixin._normalize_logits_to_keep(indices, seq_len=5) == [0, 2, 3]

    def test_tensor_negative_indices_wrap(self) -> None:
        indices = torch.tensor([-1, -3])
        assert MobilintModelMixin._normalize_logits_to_keep(indices, seq_len=5) == [2, 4]

    def test_tensor_out_of_range_indices_are_dropped(self) -> None:
        indices = torch.tensor([-100, 0, 4, 5, 99])
        assert MobilintModelMixin._normalize_logits_to_keep(indices, seq_len=5) == [0, 4]

    def test_tensor_empty_returns_empty_list(self) -> None:
        indices = torch.tensor([], dtype=torch.long)
        assert MobilintModelMixin._normalize_logits_to_keep(indices, seq_len=5) == []


# ---------------------------------------------------------------------------
# _mxq_supports_all_logits
# ---------------------------------------------------------------------------


class TestMxqSupportsAllLogits:
    def test_static_token_axis_returns_false(self) -> None:
        model = make_model(StaticLastOnlyMxq())
        assert model._mxq_supports_all_logits() is False

    def test_dynamic_token_axis_returns_true(self) -> None:
        model = make_model(DynamicAxisMxq())
        assert model._mxq_supports_all_logits() is True

    def test_result_is_cached_on_instance(self) -> None:
        mxq = StaticLastOnlyMxq()
        model = make_model(mxq)
        assert model._mxq_supports_all_logits() is False

        # Flip the underlying shape; a fresh probe would now claim dynamic.
        mxq.get_model_output_shape = lambda: [(1, -1, mxq.vocab_size)]  # type: ignore[assignment]
        # Cached: should still report False.
        assert model._mxq_supports_all_logits() is False

    def test_raising_backend_falls_back_to_false(self) -> None:
        class _ExplodingMxq(StaticLastOnlyMxq):
            def get_model_output_shape(self):  # noqa: D401
                raise RuntimeError("no shape for you")

        model = make_model(_ExplodingMxq())
        assert model._mxq_supports_all_logits() is False


# ---------------------------------------------------------------------------
# Single-item llm_forward paths
# ---------------------------------------------------------------------------


class TestLlmForwardSingle:
    def _run(
        self,
        mxq,
        seq_len: int,
        hidden_size: int,
        *,
        logits_to_keep,
        prefill_chunk_size: Optional[int] = None,
    ):
        model = make_model(mxq)
        inputs_embeds = torch.arange(seq_len * hidden_size, dtype=torch.float32).reshape(
            1, seq_len, hidden_size
        )
        cache_position = torch.arange(seq_len)
        logits = model.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            cache_position=cache_position,
            prefill_chunk_size=prefill_chunk_size,
            logits_to_keep=logits_to_keep,
        )
        return model, logits

    def test_default_keep_uses_fast_path_with_last_token_logits(self) -> None:
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        _model, logits = self._run(mxq, seq_len=6, hidden_size=3, logits_to_keep=1, prefill_chunk_size=3)

        # Two chunks of size 3 with the caller's prefill_chunk_size.
        chunk_seqs = [c["shape"][2] for c in mxq.calls]
        assert chunk_seqs == [3, 3]
        # Path 1 squeezes the outer batch axis → (1, vocab).
        assert logits.shape == (1, mxq.vocab_size)

    def test_default_keep_ignores_dynamic_axis_probe(self) -> None:
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        model, logits = self._run(mxq, seq_len=6, hidden_size=3, logits_to_keep=1, prefill_chunk_size=3)

        # is_default_keep short-circuits before probing, so no cached probe result yet.
        assert getattr(model, "_mxq_all_logits_cached", None) is None
        # Only the last chunk's logits are returned in shape (1, chunk_len, vocab)
        # → squeeze(0) → (chunk_len, vocab) because dynamic mxq returned 3d.
        assert logits.shape == (3, mxq.vocab_size)

    def test_dynamic_axis_keep_all_returns_every_position(self) -> None:
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        model, logits = self._run(
            mxq, seq_len=6, hidden_size=3, logits_to_keep=0, prefill_chunk_size=3
        )

        assert model._mxq_supports_all_logits() is True
        assert logits.shape == (6, mxq.vocab_size)

        # Reconstruct the same per-chunk payload the fake produced so we can
        # check the concatenation ordering.
        expected_chunks = []
        cache_size_running = 0
        for chunk_len in (3, 3):
            offset = cache_size_running * mxq.vocab_size
            expected_chunks.append(
                np.arange(chunk_len * mxq.vocab_size, dtype=np.float32).reshape(
                    1, chunk_len, mxq.vocab_size
                )
                + offset
            )
            cache_size_running = 0  # past_key_values is None → cache_size stays 0
        expected = np.concatenate(expected_chunks, axis=1).squeeze(0)
        np.testing.assert_allclose(logits.numpy(), expected)

    def test_dynamic_axis_keep_last_n_slices_expected_positions(self) -> None:
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        _model, logits = self._run(
            mxq, seq_len=6, hidden_size=3, logits_to_keep=2, prefill_chunk_size=3
        )
        assert logits.shape == (2, mxq.vocab_size)

    def test_dynamic_axis_tensor_indices_pick_out_positions(self) -> None:
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        indices = torch.tensor([0, 2, 5])
        _model, logits = self._run(
            mxq, seq_len=6, hidden_size=3, logits_to_keep=indices, prefill_chunk_size=3
        )
        assert logits.shape == (3, mxq.vocab_size)

    def test_fallback_interleaves_size_one_infer_for_kept_positions(self) -> None:
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        indices = torch.tensor([2, 5])
        _model, logits = self._run(
            mxq, seq_len=6, hidden_size=3, logits_to_keep=indices, prefill_chunk_size=4
        )

        # Fallback: prefill 0..2 (chunk of 2), size-1 at 2, prefill 3..5 (chunk of 2), size-1 at 5.
        # The prefix stride is clamped to the next kept position, not the caller's chunk size.
        chunk_seqs = [c["shape"][2] for c in mxq.calls]
        assert chunk_seqs == [2, 1, 2, 1]
        # ``.squeeze(0)`` in path 3 drops the leading batch axis.
        assert logits.shape == (len(indices), mxq.vocab_size)

    def test_fallback_advances_kv_cache_across_all_positions(self) -> None:
        from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache

        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        model = make_model(mxq)
        cache = MobilintCache(mxq)
        seq_len = 6
        inputs_embeds = torch.zeros(1, seq_len, 3)
        cache_position = torch.arange(seq_len)

        model.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            cache_position=cache_position,
            prefill_chunk_size=4,
            logits_to_keep=torch.tensor([2, 5]),
        )
        # KV cache must advance monotonically through every position, even when
        # only a subset yields logits.
        assert cache.get_seq_length() == seq_len

    def test_fallback_empty_tensor_returns_empty_logits(self) -> None:
        """An empty ``logits_to_keep`` tensor must not raise on np.concatenate."""
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        _model, logits = self._run(
            mxq,
            seq_len=6,
            hidden_size=3,
            logits_to_keep=torch.tensor([], dtype=torch.long),
            prefill_chunk_size=4,
        )
        # No kept positions → no size-1 infer calls, but the KV prefix loop
        # still walks the whole sequence in normal-sized chunks.
        chunk_seqs = [c["shape"][2] for c in mxq.calls]
        assert chunk_seqs == [4, 2]
        assert logits.shape == (0, 0)

    def test_fallback_all_out_of_range_indices_returns_empty_logits(self) -> None:
        """All-out-of-range ``logits_to_keep`` must not raise on np.concatenate."""
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        _model, logits = self._run(
            mxq,
            seq_len=6,
            hidden_size=3,
            logits_to_keep=torch.tensor([100, -100]),
            prefill_chunk_size=4,
        )
        chunk_seqs = [c["shape"][2] for c in mxq.calls]
        assert chunk_seqs == [4, 2]
        assert logits.shape == (0, 0)

    def test_fallback_empty_tensor_still_advances_kv_cache(self) -> None:
        from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache

        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        model = make_model(mxq)
        cache = MobilintCache(mxq)
        seq_len = 6
        inputs_embeds = torch.zeros(1, seq_len, 3)
        cache_position = torch.arange(seq_len)

        model.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            cache_position=cache_position,
            prefill_chunk_size=4,
            logits_to_keep=torch.tensor([], dtype=torch.long),
        )
        # Even with no kept positions, the KV cache must be fully advanced so
        # subsequent decode steps observe the same cache state as the other
        # branches would leave behind.
        assert cache.get_seq_length() == seq_len


# ---------------------------------------------------------------------------
# Batched _llm_forward_batch paths
# ---------------------------------------------------------------------------


class TestLlmForwardBatch:
    def _run(
        self,
        mxq,
        *,
        attention_mask: torch.Tensor,
        hidden_size: int,
        logits_to_keep,
        prefill_chunk_size: Optional[int] = None,
        max_batch_size: int = 4,
    ):
        model = make_model(mxq, max_batch_size=max_batch_size)
        batch, seq_len = attention_mask.shape
        inputs_embeds = torch.arange(batch * seq_len * hidden_size, dtype=torch.float32).reshape(
            batch, seq_len, hidden_size
        )
        cache_position = torch.arange(seq_len)
        logits = model.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            cache_position=cache_position,
            prefill_chunk_size=prefill_chunk_size,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
        )
        return model, logits

    def test_default_keep_path_preserves_existing_shape(self) -> None:
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=2)
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
        _model, logits = self._run(
            mxq, attention_mask=attention_mask, hidden_size=4, logits_to_keep=1, prefill_chunk_size=2
        )
        assert logits.shape == (2, 1, mxq.vocab_size)

    def test_dynamic_axis_keep_all_returns_per_item_padded_stack(self) -> None:
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=0,
            prefill_chunk_size=3,
        )
        # Both items pad up to the longest kept-length (3 positions for item 1;
        # item 0 has 2 real positions and is right-padded).
        assert logits.shape == (2, 3, mxq.vocab_size)
        # The padded tail of item 0 must be -inf so softmax zeros it out.
        assert torch.all(torch.isinf(logits[0, 2]) & (logits[0, 2] < 0))

    def test_dynamic_axis_zero_length_row_is_softmax_masked(self) -> None:
        """A batch item whose attention_mask row is entirely zero must not crash
        Path 2 on ``np.concatenate([])`` — Path 3 already tolerates this and
        Path 2 should be symmetric. The zero-length row's slot in the stacked
        output is right-padded with -inf so softmax assigns zero probability.
        """
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=0,
            prefill_chunk_size=3,
        )
        assert logits.shape == (2, 3, mxq.vocab_size)
        assert torch.all(torch.isinf(logits[0]) & (logits[0] < 0))

    def test_dynamic_axis_tensor_indices_apply_per_item(self) -> None:
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long)
        indices = torch.tensor([0, 2])
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=indices,
            prefill_chunk_size=4,
        )
        # Both items have position 0 and 2 in range → 2 kept positions each.
        assert logits.shape == (2, 2, mxq.vocab_size)

    def test_fallback_interleaves_prefill_and_size_one_across_batch(self) -> None:
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long)
        # Request positions 2 and 3 so the fallback branch runs; logits_to_keep=1
        # would hit the fast path and skip the fallback entirely.
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=torch.tensor([2, 3]),
            prefill_chunk_size=2,
        )
        # Both items keep positions {2, 3}, so min_keep_start=2. First chunk
        # advances cursor 0→2 with the caller's prefill_chunk_size, then chunks
        # drop to 1 to capture positions 2 and 3.
        chunk_widths = [c["batch"][0][1] for c in mxq.calls]
        assert chunk_widths == [2, 1, 1]
        assert logits.shape == (2, 2, mxq.vocab_size)

    def test_fallback_handles_variable_per_item_kept_positions(self) -> None:
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        # Item 0 has 3 real tokens; item 1 has 4. Keep-all on both yields
        # 3 and 4 positions respectively → padding on item 0.
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long)
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=0,
            prefill_chunk_size=2,
        )
        assert logits.shape == (2, 4, mxq.vocab_size)
        # Item 0's slot 3 is padding — filled with -inf so softmax masks it.
        assert torch.all(torch.isinf(logits[0, 3]) & (logits[0, 3] < 0))

    def test_fallback_asymmetric_batch_avoids_size_one_for_exhausted_item(self) -> None:
        """One item has its only kept position at 0, the other has kept at 5.

        With the previous batch-wide ``min_keep_start`` heuristic, every cursor
        step after position 0 would be forced to chunk=1 — even though item 1
        was done capturing after step 0 and item 0 didn't need another capture
        until position 5. The per-item pointer lets the middle walk use the
        caller's ``prefill_chunk_size``.
        """
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        # Item 0 seq_len=6 keeps {0, 5}; item 1 seq_len=1 keeps {0} (5 clamped away).
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]], dtype=torch.long
        )
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=torch.tensor([0, 5]),
            prefill_chunk_size=4,
        )
        # Expected trace: (chunk=1 at cursor 0 for both items), (chunk=4 for
        # item 0 only, walking 1→5), (chunk=1 at cursor 5 for item 0).
        assert len(mxq.calls) == 3
        assert mxq.calls[0]["batch"] == [(0, 1, 0), (1, 1, 0)]
        assert mxq.calls[1]["batch"] == [(0, 4, 0)]
        assert mxq.calls[2]["batch"] == [(0, 1, 0)]

        # Item 0 keeps 2 positions; item 1 keeps 1 → padded stack (2, 2, vocab).
        assert logits.shape == (2, 2, mxq.vocab_size)
        # Item 1's slot 1 is padding — -inf so softmax masks it.
        assert torch.all(torch.isinf(logits[1, 1]) & (logits[1, 1] < 0))

    def test_fallback_disjoint_per_item_kept_positions(self) -> None:
        """Kept sets share no positions: item 0 keeps {1}, item 1 keeps {5}.

        Previously ``min_keep_start=1`` forced chunk=1 across the entire walk
        after cursor=1, even though item 0 was inactive from cursor=2 onward
        and item 1 didn't need another capture until 5.
        """
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        # Item 0 seq_len=2 keeps {1}; item 1 seq_len=6 keeps {5} via last-index.
        attention_mask = torch.tensor(
            [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long
        )
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=torch.tensor([-1]),
            prefill_chunk_size=4,
        )
        # Trace: chunk=1 @0 (both, no capture), chunk=1 @1 (both, capture item 0),
        # chunk=3 @2 (item 1 only, KV walk), chunk=1 @5 (item 1, capture).
        per_call = [(len(c["batch"]), c["batch"][0][1]) for c in mxq.calls]
        assert per_call == [(2, 1), (2, 1), (1, 3), (1, 1)]

        # Item 1 must not appear in the middle KV-only walk except by itself.
        assert mxq.calls[2]["batch"] == [(1, 3, 0)]

        assert logits.shape == (2, 1, mxq.vocab_size)
