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

import warnings
from typing import Optional

import numpy as np
import pytest
import torch

from mblt_model_zoo.hf_transformers.utils import modeling_utils
from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin
from tests.transformers._fake_mxq import (
    DynamicAxisMxq,
    StaticLastOnlyMxq,
    make_model,
)


# ---------------------------------------------------------------------------
# _output_positions_for_logits_to_keep / _walk_positions_for_logits_to_keep
# ---------------------------------------------------------------------------


class TestOutputPositionsForLogitsToKeep:
    """HF-compatible selector: caller order preserved, duplicates kept."""

    def test_int_zero_keeps_every_position(self) -> None:
        assert MobilintModelMixin._output_positions_for_logits_to_keep(0, 5) == [0, 1, 2, 3, 4]

    def test_int_one_keeps_last_position(self) -> None:
        assert MobilintModelMixin._output_positions_for_logits_to_keep(1, 5) == [4]

    def test_int_n_keeps_last_n_positions(self) -> None:
        assert MobilintModelMixin._output_positions_for_logits_to_keep(3, 5) == [2, 3, 4]

    def test_int_oversized_keeps_all_positions(self) -> None:
        assert MobilintModelMixin._output_positions_for_logits_to_keep(9, 4) == [0, 1, 2, 3]

    def test_int_equal_seq_len_keeps_all_positions(self) -> None:
        assert MobilintModelMixin._output_positions_for_logits_to_keep(5, 5) == [0, 1, 2, 3, 4]

    def test_int_negative_raises_value_error(self) -> None:
        # Negative ints are a caller mistake (HF doesn't accept them and the
        # tensor-negative-wrap path is a separate code path). Reject explicitly
        # rather than silently mapping to keep-all, which the initial
        # implementation in commit d92a353 did.
        with pytest.raises(ValueError, match=r"logits_to_keep must be a non-negative int.*got -1"):
            MobilintModelMixin._output_positions_for_logits_to_keep(-1, 3)

    def test_int_negative_raises_via_llm_forward(self) -> None:
        model = make_model(StaticLastOnlyMxq(vocab_size=5, max_width=4))
        inputs_embeds = torch.zeros(1, 4, 3)
        cache_position = torch.arange(4)
        with pytest.raises(ValueError, match=r"logits_to_keep must be a non-negative int.*got -2"):
            model.llm_forward(
                inputs_embeds=inputs_embeds,
                past_key_values=None,
                cache_position=cache_position,
                prefill_chunk_size=4,
                logits_to_keep=-2,
            )

    def test_tensor_preserves_caller_order(self) -> None:
        indices = torch.tensor([3, 0, 3, 2])
        assert MobilintModelMixin._output_positions_for_logits_to_keep(indices, seq_len=5) == [3, 0, 3, 2]

    def test_tensor_preserves_duplicates(self) -> None:
        indices = torch.tensor([2, 2])
        assert MobilintModelMixin._output_positions_for_logits_to_keep(indices, seq_len=5) == [2, 2]

    def test_tensor_negative_indices_wrap(self) -> None:
        indices = torch.tensor([-1, -3])
        assert MobilintModelMixin._output_positions_for_logits_to_keep(indices, seq_len=5) == [4, 2]

    def test_tensor_out_of_range_index_raises(self) -> None:
        indices = torch.tensor([0, 4, 99])
        with pytest.raises(IndexError, match=r"99.*seq_len=5"):
            MobilintModelMixin._output_positions_for_logits_to_keep(indices, seq_len=5)

    def test_tensor_negative_out_of_range_index_raises(self) -> None:
        # -100 wraps to -95, which is still < 0 and must not be silently dropped.
        indices = torch.tensor([-100])
        with pytest.raises(IndexError, match=r"-100.*seq_len=5"):
            MobilintModelMixin._output_positions_for_logits_to_keep(indices, seq_len=5)

    def test_tensor_empty_returns_empty_list(self) -> None:
        indices = torch.tensor([], dtype=torch.long)
        assert MobilintModelMixin._output_positions_for_logits_to_keep(indices, seq_len=5) == []


class TestWalkPositionsForLogitsToKeep:
    """Cursor helper: sorted-unique for monotonic KV walks."""

    def test_tensor_sorts_and_dedupes(self) -> None:
        indices = torch.tensor([3, 0, 3, 2])
        assert MobilintModelMixin._walk_positions_for_logits_to_keep(indices, seq_len=5) == [0, 2, 3]

    def test_int_matches_output_helper(self) -> None:
        # For int inputs the two helpers are structurally the same because
        # int semantics never produce duplicates or unsorted output.
        assert MobilintModelMixin._walk_positions_for_logits_to_keep(3, 5) == [2, 3, 4]
        assert MobilintModelMixin._walk_positions_for_logits_to_keep(0, 4) == [0, 1, 2, 3]

    def test_tensor_negative_indices_wrap(self) -> None:
        indices = torch.tensor([-1, -3])
        assert MobilintModelMixin._walk_positions_for_logits_to_keep(indices, seq_len=5) == [2, 4]

    def test_tensor_out_of_range_index_raises(self) -> None:
        indices = torch.tensor([0, 99])
        with pytest.raises(IndexError, match=r"99.*seq_len=5"):
            MobilintModelMixin._walk_positions_for_logits_to_keep(indices, seq_len=5)


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
        import qbruntime

        class _ExplodingMxq(StaticLastOnlyMxq):
            def get_model_output_shape(self):  # noqa: D401
                raise qbruntime.QbRuntimeError("no shape for you")

        model = make_model(_ExplodingMxq())
        assert model._mxq_supports_all_logits() is False

    def test_probe_honors_patched_sentinel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The dynamic-axis sentinel is a module-level constant, not a hard-coded literal.

        If a future backend represents "dynamic" with a different marker we
        can retarget the probe by rebinding ``_MXQ_DYNAMIC_AXIS_SENTINEL``.
        """
        monkeypatch.setattr(modeling_utils, "_MXQ_DYNAMIC_AXIS_SENTINEL", -2)

        class _NewSentinelMxq(StaticLastOnlyMxq):
            def get_model_output_shape(self):
                return [(1, -2, self.vocab_size)]

        model = make_model(_NewSentinelMxq())
        assert model._mxq_supports_all_logits() is True

        # The old ``-1`` sentinel must no longer trigger the dynamic path.
        stale = make_model(DynamicAxisMxq())
        assert stale._mxq_supports_all_logits() is False


# ---------------------------------------------------------------------------
# _mxq_static_vocab_size
# ---------------------------------------------------------------------------


class TestMxqStaticVocabSize:
    def test_first_call_returns_vocab_size(self) -> None:
        mxq = StaticLastOnlyMxq(vocab_size=7)
        model = make_model(mxq)
        assert model._mxq_static_vocab_size() == 7

    def test_result_is_cached_on_instance(self) -> None:
        mxq = StaticLastOnlyMxq(vocab_size=5)
        model = make_model(mxq)
        assert model._mxq_static_vocab_size() == 5

        # Flip the backend to a different vocab dim; a fresh probe would now
        # return 9. The cached value must win — the compiled vocab dim is a
        # fixed model property so we never want to re-probe.
        mxq.get_model_output_shape = lambda: [(1, 1, 9)]  # type: ignore[assignment]
        assert model._mxq_static_vocab_size() == 5

    def test_raising_backend_propagates_error(self) -> None:
        import qbruntime

        class _ExplodingMxq(StaticLastOnlyMxq):
            def get_model_output_shape(self):  # noqa: D401
                raise qbruntime.QbRuntimeError("no shape for you")

        model = make_model(_ExplodingMxq())
        # Contract: probe failure re-raises rather than returning a sentinel.
        # Path 2 has no valid fallback for an unknown vocab dim.
        with pytest.raises(qbruntime.QbRuntimeError, match="no shape for you"):
            model._mxq_static_vocab_size()
        # Nothing was cached, so a subsequent call re-probes and re-raises.
        assert getattr(model, "_mxq_static_vocab_cached", None) is None


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

    def test_scalar_tensor_seq_len_minus_one_uses_fast_path(self) -> None:
        """``torch.tensor([seq_len - 1])`` is the tensor equivalent of ``logits_to_keep=1``.

        The dynamic-axis probe would set ``_mxq_all_logits_cached`` on the
        instance; Path 1 short-circuits before probing, so its absence
        after the call proves the shortcut fired.
        """
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=6)
        seq_len = 6
        model, logits = self._run(
            mxq,
            seq_len=seq_len,
            hidden_size=3,
            logits_to_keep=torch.tensor([seq_len - 1]),
            prefill_chunk_size=seq_len,
        )
        assert getattr(model, "_mxq_all_logits_cached", None) is None
        chunk_seqs = [c["shape"][2] for c in mxq.calls]
        assert chunk_seqs == [seq_len]
        assert logits.shape == (1, mxq.vocab_size)

    def test_scalar_tensor_negative_one_uses_fast_path(self) -> None:
        """``torch.tensor([-1])`` is the HF negative-wrap equivalent of ``[seq_len - 1]``."""
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=6)
        seq_len = 6
        model, logits = self._run(
            mxq,
            seq_len=seq_len,
            hidden_size=3,
            logits_to_keep=torch.tensor([-1]),
            prefill_chunk_size=seq_len,
        )
        assert getattr(model, "_mxq_all_logits_cached", None) is None
        chunk_seqs = [c["shape"][2] for c in mxq.calls]
        assert chunk_seqs == [seq_len]
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
        # check the concatenation ordering. Without a caller cache, ``_do_infer``
        # uses ``start`` as the running cache_size so the NPU sees prior chunks'
        # KV state — the fake mirrors that by offsetting per chunk by cache_size.
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
            cache_size_running += chunk_len
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

    def test_dynamic_axis_tensor_preserves_caller_order_and_duplicates(self) -> None:
        """HF fancy-indexing preserves caller order and duplicates; Path 2 mirrors that.

        Building the "expected" from the fake's own per-chunk payload keeps
        the assertion tied to observable behavior rather than the fake's
        internal layout.
        """
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        seq_len = 5
        indices = torch.tensor([3, 0, 3, 2])
        _model, logits = self._run(
            mxq, seq_len=seq_len, hidden_size=3, logits_to_keep=indices, prefill_chunk_size=seq_len
        )

        # Reconstruct the single (chunk_len=5) payload the fake produced.
        full = np.arange(seq_len * mxq.vocab_size, dtype=np.float32).reshape(seq_len, mxq.vocab_size)
        expected = full[[3, 0, 3, 2], :]
        assert logits.shape == (4, mxq.vocab_size)
        np.testing.assert_allclose(logits.numpy(), expected)
        # Duplicate index 3 → rows 0 and 2 are identical.
        np.testing.assert_array_equal(logits[0].numpy(), logits[2].numpy())

    def test_dynamic_axis_tensor_negative_indices_pick_positions_in_order(self) -> None:
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        seq_len = 4
        _model, logits = self._run(
            mxq,
            seq_len=seq_len,
            hidden_size=3,
            logits_to_keep=torch.tensor([-1, -2]),
            prefill_chunk_size=seq_len,
        )

        full = np.arange(seq_len * mxq.vocab_size, dtype=np.float32).reshape(seq_len, mxq.vocab_size)
        expected = full[[3, 2], :]  # -1 → 3, -2 → 2
        np.testing.assert_allclose(logits.numpy(), expected)

    def test_dynamic_axis_tensor_out_of_range_raises(self) -> None:
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        with pytest.raises(IndexError, match=r"99.*seq_len=5"):
            self._run(
                mxq,
                seq_len=5,
                hidden_size=3,
                logits_to_keep=torch.tensor([0, 4, 99]),
                prefill_chunk_size=5,
            )

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

    def test_fallback_advances_cache_size_without_caller_cache(self) -> None:
        """Path 3 with ``past_key_values=None`` must still tell the NPU about
        prior tokens.

        Without this, the discarded prefix ``_do_infer`` calls would pass
        ``cache_size=0`` and each size-1 kept-position capture would be
        evaluated as if it were the start of a fresh sequence — silently
        corrupting keep-all / perplexity logits when ``.forward()`` is
        called without ``use_cache=True`` (the new HF default is
        ``logits_to_keep=0``, which triggers Path 3 on last-only MXQ).
        """
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        indices = torch.tensor([2, 5])
        _model, _logits = self._run(
            mxq, seq_len=6, hidden_size=3, logits_to_keep=indices, prefill_chunk_size=4
        )
        # Chunks: (0..2 prefill), (2..3 capture), (3..5 prefill), (5..6 capture).
        # cache_size at each infer entry equals the running "processed so far"
        # count, which is the chunk's ``start`` position.
        cache_sizes = [c["cache_size"] for c in mxq.calls]
        assert cache_sizes == [0, 2, 3, 5]

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

    def test_fallback_out_of_range_indices_raise_indexerror(self) -> None:
        """HF fancy-indexing raises on out-of-range indices; we mirror that."""
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        with pytest.raises(IndexError, match=r"100.*seq_len=6"):
            self._run(
                mxq,
                seq_len=6,
                hidden_size=3,
                logits_to_keep=torch.tensor([100, -100]),
                prefill_chunk_size=4,
            )

    def test_fallback_tensor_preserves_caller_order_and_duplicates(self) -> None:
        """Path 3 KV walk is sorted-unique; final output re-uses positions in caller order.

        The KV cache is threaded through so ``cache_size`` (the fake fill) is
        distinct at each captured position: size-1 infer at position p (with
        prior positions already walked) returns ``full(vocab, p)``. That makes
        the picked-per-position identity visible in the assembled tensor.
        """
        from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache

        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        model = make_model(mxq)
        cache = MobilintCache(mxq)
        seq_len = 5
        inputs_embeds = torch.zeros(1, seq_len, 3)
        cache_position = torch.arange(seq_len)

        indices = torch.tensor([3, 0, 3, 2])
        logits = model.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            cache_position=cache_position,
            prefill_chunk_size=4,
            logits_to_keep=indices,
        )

        assert logits.shape == (4, mxq.vocab_size)
        # Duplicate index 3 → rows 0 and 2 pick the same per_position tensor.
        np.testing.assert_array_equal(logits[0].numpy(), logits[2].numpy())
        # Fill at row 0 should match cache_size at position 3 (i.e. 3.0),
        # row 1 → position 0 (fill=0), row 3 → position 2 (fill=2).
        assert float(logits[0, 0]) == 3.0
        assert float(logits[1, 0]) == 0.0
        assert float(logits[3, 0]) == 2.0
        # KV walk hit every position exactly once.
        assert cache.get_seq_length() == seq_len

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

    def test_mixed_length_scalar_tensor_does_not_take_shortcut(self) -> None:
        """A scalar tensor selector cannot uniformly represent "last position"
        across a mixed-length batch, so Path 1 must not fire.

        For lengths ``[2, 3]`` with ``torch.tensor([-1])`` the fallback
        walks per-item and captures position 1 for item 0 and position 2
        for item 1. The trace-visible tell that Path 1 was skipped is the
        size-1 batched infer calls; Path 1 would have used the caller's
        ``prefill_chunk_size`` for a single wider infer instead.
        """
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=torch.tensor([-1]),
            prefill_chunk_size=4,
        )
        per_call_widths = [c["batch"][0][1] for c in mxq.calls]
        assert per_call_widths == [1, 1, 1]
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

    def test_dynamic_axis_zero_length_row_raises_value_error(self) -> None:
        """A batch item whose attention_mask row is entirely zero is now a hard
        error under Path 2, matching Path 1's long-standing behavior. Previously
        Path 2 silently right-padded such rows with -inf, so the same batch
        would fail on ``logits_to_keep=1`` (the ``.generate()`` default) but
        pass on ``logits_to_keep=0`` — a footgun the unified check closes.
        """
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
        with pytest.raises(ValueError, match=r"Zero-length rows"):
            self._run(
                mxq,
                attention_mask=attention_mask,
                hidden_size=4,
                logits_to_keep=0,
                prefill_chunk_size=3,
            )

    @pytest.mark.parametrize(
        "mxq_cls,logits_to_keep",
        [
            (DynamicAxisMxq, 0),
            (StaticLastOnlyMxq, torch.tensor([2, 3])),
        ],
        ids=["path2_dynamic_axis_keep_all", "path3_last_only_tensor_selector"],
    )
    def test_zero_length_row_raises_value_error_across_all_paths(
        self, mxq_cls, logits_to_keep
    ) -> None:
        """Lock in the unified zero-length contract: the check fires before any
        is_default_keep / supports_all decision, so Path 2 (dynamic-axis with
        keep-all) and Path 3 (last-only with tensor selector) both raise —
        not just Path 1. Prevents the asymmetry from silently regressing if
        the check drifts back into any single path.
        """
        mxq = mxq_cls(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.long)
        with pytest.raises(ValueError, match=r"Zero-length rows"):
            self._run(
                mxq,
                attention_mask=attention_mask,
                hidden_size=4,
                logits_to_keep=logits_to_keep,
                prefill_chunk_size=4,
            )

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

    def test_dynamic_axis_tensor_preserves_caller_order_and_duplicates_per_item(self) -> None:
        """Path 2 batched: caller order and duplicates survive per batch item.

        With item lengths 4 and 2, ``logits_to_keep=[0, -1]`` resolves to
        [0, 3] for item 0 and [0, 1] for item 1 — same kept length so no
        padding, but the negative-wrap acts per-item.
        """
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long)
        indices = torch.tensor([0, -1])
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=indices,
            prefill_chunk_size=4,
        )
        assert logits.shape == (2, 2, mxq.vocab_size)

    def test_dynamic_axis_tensor_duplicates_produce_repeated_rows(self) -> None:
        """A tensor with an explicit duplicate index emits the same row twice per item."""
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
        indices = torch.tensor([2, 0, 2])
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=indices,
            prefill_chunk_size=3,
        )
        assert logits.shape == (2, 3, mxq.vocab_size)
        # For every item, row 0 (index 2) equals row 2 (also index 2).
        for j in range(2):
            np.testing.assert_array_equal(logits[j, 0].numpy(), logits[j, 2].numpy())

    def test_fallback_tensor_duplicates_produce_repeated_rows_per_item(self) -> None:
        """Path 3 batched: walk-positions drive the cursor, output-positions reassemble."""
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
        indices = torch.tensor([2, 0, 2])
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=indices,
            prefill_chunk_size=4,
        )
        assert logits.shape == (2, 3, mxq.vocab_size)
        # Duplicate index 2 must yield identical rows within each batch item.
        for j in range(2):
            np.testing.assert_array_equal(logits[j, 0].numpy(), logits[j, 2].numpy())

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
        """One item's walk-set is exhausted after cursor 0; the other keeps going to 5.

        With the previous batch-wide ``min_keep_start`` heuristic, every cursor
        step after position 0 would be forced to chunk=1 — even though item 1
        was done capturing after step 0 and item 0 didn't need another capture
        until position 5. The per-item pointer lets the middle walk use the
        caller's ``prefill_chunk_size``.

        ``[0, -1]`` picks position 0 and the last valid position per item —
        item 0 (seq_len=6) resolves to [0, 5]; item 1 (seq_len=1) resolves to
        [0, 0] (walk-set collapses to {0} after dedup).
        """
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]], dtype=torch.long
        )
        _model, logits = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=torch.tensor([0, -1]),
            prefill_chunk_size=4,
        )
        # Expected trace: (chunk=1 at cursor 0 for both items), (chunk=4 for
        # item 0 only, walking 1→5), (chunk=1 at cursor 5 for item 0).
        # cache_sizes track the running cursor per item (start=cursor when
        # past_key_values is None) so the NPU sees prior chunks' KV state.
        assert len(mxq.calls) == 3
        assert mxq.calls[0]["batch"] == [(0, 1, 0), (1, 1, 0)]
        assert mxq.calls[1]["batch"] == [(0, 4, 1)]
        assert mxq.calls[2]["batch"] == [(0, 1, 5)]

        # Both items keep 2 positions (item 1's second row repeats position 0).
        assert logits.shape == (2, 2, mxq.vocab_size)
        # Item 1's duplicate index means row 0 == row 1 (both pick position 0).
        np.testing.assert_array_equal(logits[1, 0].numpy(), logits[1, 1].numpy())

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
        # cache_size=2 because item 1 has been walked contiguously through
        # cursor 0 and 1 by prior chunks (running cursor when the caller
        # passes past_key_values=None).
        assert mxq.calls[2]["batch"] == [(1, 3, 2)]

        assert logits.shape == (2, 1, mxq.vocab_size)


# ---------------------------------------------------------------------------
# Batched empty-selection early return
# ---------------------------------------------------------------------------


class TestBatchedEmptySelection:
    """Path 2 and Path 3 short-circuit to a ``(batch, 0, 0)`` tensor when every
    batch item has an empty selection.

    Rationale: the padding loops in both paths build ``torch.full((rows, cols), -inf)``
    tensors that happen to be benign when ``max_keep_len == 0`` (the ``item.shape[0]
    < max_keep_len`` guard skips the fill). That reliance is fragile — a future
    change to how ``max_keep_len`` or ``vocab_size`` is derived could break silently
    — so both paths return the empty-shape contract explicitly instead.

    Post-#1 (``_wrap_and_validate_indices`` raises on out-of-range), the only
    caller-facing way to make every batch item land in this branch is an empty
    ``logits_to_keep`` tensor. Out-of-range test scenarios like ``torch.tensor([100, 200])``
    would raise ``IndexError`` before reaching the early return, so no dedicated
    "all-out-of-range" test is included.
    """

    def _run(
        self,
        mxq,
        *,
        attention_mask: torch.Tensor,
        hidden_size: int,
        logits_to_keep,
        prefill_chunk_size: Optional[int] = None,
        max_batch_size: int = 4,
        dtype: torch.dtype = torch.float32,
    ):
        model = make_model(mxq, max_batch_size=max_batch_size)
        batch, seq_len = attention_mask.shape
        inputs_embeds = torch.arange(batch * seq_len * hidden_size, dtype=dtype).reshape(
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
        return model, logits, inputs_embeds

    def test_batched_all_empty_tensor_returns_zero_shape_dynamic_axis(self) -> None:
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
        _model, logits, inputs_embeds = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=torch.tensor([], dtype=torch.long),
            prefill_chunk_size=3,
        )
        assert logits.shape == (2, 0, 0)
        assert logits.dtype == inputs_embeds.dtype
        assert logits.device == inputs_embeds.device

    def test_batched_all_empty_tensor_returns_zero_shape_last_only(self) -> None:
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
        _model, logits, inputs_embeds = self._run(
            mxq,
            attention_mask=attention_mask,
            hidden_size=4,
            logits_to_keep=torch.tensor([], dtype=torch.long),
            prefill_chunk_size=3,
        )
        assert logits.shape == (2, 0, 0)
        assert logits.dtype == inputs_embeds.dtype
        assert logits.device == inputs_embeds.device

    def test_batched_mixed_empty_and_nonempty_raises_value_error_dynamic_axis(
        self,
    ) -> None:
        """Zero-length rows are rejected up front regardless of selector shape:
        the empty-selector early return must NOT swallow the zero-length
        contract. Item 0's attention_mask row is zero; item 1 has 3 real
        positions — Path 2 must raise before considering the selector.
        """
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
        with pytest.raises(ValueError, match=r"Zero-length rows"):
            self._run(
                mxq,
                attention_mask=attention_mask,
                hidden_size=4,
                logits_to_keep=0,
                prefill_chunk_size=3,
            )

    def test_batched_mixed_empty_and_nonempty_raises_value_error_last_only(
        self,
    ) -> None:
        """Same guard as above, but for Path 3 (last-only MXQ fallback)."""
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        attention_mask = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
        with pytest.raises(ValueError, match=r"Zero-length rows"):
            self._run(
                mxq,
                attention_mask=attention_mask,
                hidden_size=4,
                logits_to_keep=0,
                prefill_chunk_size=3,
            )


# ---------------------------------------------------------------------------
# One-shot slow-path warning on last-only MXQ + non-default logits_to_keep
# ---------------------------------------------------------------------------


class TestLastOnlySlowPathWarning:
    """Path 3 fires a ``UserWarning`` once per instance so manual .forward()
    callers see the per-token infer cost without having to read source.

    The switch from ``logger.warning`` to ``warnings.warn`` is the reason
    these tests use ``warnings.catch_warnings`` instead of ``caplog``: the
    point of the change is that the warning's reported location is the
    caller's, not this module — a property only visible through the
    ``warnings`` machinery.
    """

    @staticmethod
    def _slow_path_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
        return [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "logits_to_keep" in str(w.message)
        ]

    def _run_single(
        self,
        model,
        *,
        seq_len: int,
        logits_to_keep,
        hidden_size: int = 3,
        prefill_chunk_size: int = 4,
    ) -> None:
        inputs_embeds = torch.arange(seq_len * hidden_size, dtype=torch.float32).reshape(
            1, seq_len, hidden_size
        )
        cache_position = torch.arange(seq_len)
        model.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            cache_position=cache_position,
            prefill_chunk_size=prefill_chunk_size,
            logits_to_keep=logits_to_keep,
        )

    def _run_batched(
        self,
        model,
        *,
        attention_mask: torch.Tensor,
        logits_to_keep,
        hidden_size: int = 4,
        prefill_chunk_size: int = 4,
    ) -> None:
        batch, seq_len = attention_mask.shape
        inputs_embeds = torch.arange(batch * seq_len * hidden_size, dtype=torch.float32).reshape(
            batch, seq_len, hidden_size
        )
        cache_position = torch.arange(seq_len)
        model.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            cache_position=cache_position,
            prefill_chunk_size=prefill_chunk_size,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
        )

    def test_warns_once_on_first_path3_entry_single_input(self) -> None:
        model = make_model(StaticLastOnlyMxq(vocab_size=5, max_width=4))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_single(model, seq_len=6, logits_to_keep=0)
            after_first = len(self._slow_path_warnings(caught))
            self._run_single(model, seq_len=6, logits_to_keep=0)
            after_second = len(self._slow_path_warnings(caught))

        assert after_first == 1
        assert after_second == 1
        message = str(self._slow_path_warnings(caught)[0].message)
        assert "logits_to_keep" in message

    def test_warns_once_on_first_path3_entry_batched(self) -> None:
        model = make_model(StaticLastOnlyMxq(vocab_size=5, max_width=4), max_batch_size=4)
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_batched(
                model, attention_mask=attention_mask, logits_to_keep=torch.tensor([2, 3])
            )
            after_first = len(self._slow_path_warnings(caught))
            self._run_batched(
                model, attention_mask=attention_mask, logits_to_keep=torch.tensor([2, 3])
            )
            after_second = len(self._slow_path_warnings(caught))

        assert after_first == 1
        assert after_second == 1
        assert "logits_to_keep" in str(self._slow_path_warnings(caught)[0].message)

    def test_no_warning_on_fast_path(self) -> None:
        single_model = make_model(StaticLastOnlyMxq(vocab_size=5, max_width=4))
        batched_model = make_model(
            StaticLastOnlyMxq(vocab_size=5, max_width=4), max_batch_size=4
        )
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_single(single_model, seq_len=6, logits_to_keep=1)
            self._run_batched(batched_model, attention_mask=attention_mask, logits_to_keep=1)

        assert self._slow_path_warnings(caught) == []

    def test_no_warning_on_dynamic_axis(self) -> None:
        single_model = make_model(DynamicAxisMxq(vocab_size=5, max_width=4))
        batched_model = make_model(
            DynamicAxisMxq(vocab_size=5, max_width=4), max_batch_size=4
        )
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_single(single_model, seq_len=6, logits_to_keep=0)
            self._run_batched(batched_model, attention_mask=attention_mask, logits_to_keep=0)

        assert self._slow_path_warnings(caught) == []

    def test_warning_is_per_instance(self) -> None:
        model_a = make_model(StaticLastOnlyMxq(vocab_size=5, max_width=4))
        model_b = make_model(StaticLastOnlyMxq(vocab_size=5, max_width=4))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_single(model_a, seq_len=6, logits_to_keep=0)
            self._run_single(model_b, seq_len=6, logits_to_keep=0)

        # One warning per instance — the flag lives on the model, not the class.
        assert len(self._slow_path_warnings(caught)) == 2

    def test_warning_stacklevel_points_past_llm_forward(self) -> None:
        """``stacklevel=4`` skips this helper, the Path 3 site, and ``llm_forward``
        so the warning is attributed to the caller of ``llm_forward`` — i.e.,
        this test method's own frame, not ``modeling_utils.py``.
        """
        model = make_model(StaticLastOnlyMxq(vocab_size=5, max_width=4))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_single(model, seq_len=6, logits_to_keep=0)

        slow = self._slow_path_warnings(caught)
        assert len(slow) == 1
        # ``_run_single`` calls ``model.llm_forward`` directly, so the
        # stacklevel=4 warning must land inside this test file, not the
        # library module.
        assert slow[0].filename.endswith("test_logits_to_keep.py")


# ---------------------------------------------------------------------------
# Regression: batched tensor selector must hoist the device->host copy
# ---------------------------------------------------------------------------


class TestBatchCpuCopyHoist:
    """Lock in the tensor-selector CPU-copy hoist and the ``lens_equal`` shared
    result in :meth:`MobilintModelMixin._llm_forward_batch`.

    The batch loop must not call ``logits_to_keep.detach().to("cpu").flatten()
    .tolist()`` per batch item — one copy up front, then per-``seq_len``
    wrap/validate via :meth:`_output_positions_from_cpu_indices`. When every
    item shares the same length the result is computed once and shared.
    """

    @staticmethod
    def _spy_from_cpu_indices(monkeypatch):
        calls: list[tuple[int, int]] = []
        orig_fn = MobilintModelMixin._output_positions_from_cpu_indices.__func__

        @classmethod
        def spy(cls, cpu_indices, seq_len):
            calls.append((id(cpu_indices), seq_len))
            return orig_fn(cls, cpu_indices, seq_len)

        monkeypatch.setattr(
            MobilintModelMixin, "_output_positions_from_cpu_indices", spy
        )
        return calls

    @staticmethod
    def _run_batched(model, *, attention_mask, logits_to_keep, hidden_size=4, prefill_chunk_size=4):
        batch, seq_len = attention_mask.shape
        inputs_embeds = torch.arange(batch * seq_len * hidden_size, dtype=torch.float32).reshape(
            batch, seq_len, hidden_size
        )
        cache_position = torch.arange(seq_len)
        return model.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            cache_position=cache_position,
            prefill_chunk_size=prefill_chunk_size,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
        )

    def test_from_cpu_indices_unit_matches_full_helper(self) -> None:
        """The new entry point resolves the same way as the tensor branch of
        :meth:`_output_positions_for_logits_to_keep`, given a pre-materialized
        CPU list — including HF-style negative wrap and range check.
        """
        assert MobilintModelMixin._output_positions_from_cpu_indices([0, -1, 2], seq_len=5) == [0, 4, 2]
        with pytest.raises(IndexError, match=r"7.*seq_len=5"):
            MobilintModelMixin._output_positions_from_cpu_indices([0, 7], seq_len=5)

    def test_lens_equal_calls_helper_exactly_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uniform-length batch + tensor selector: the resolver runs once and
        every batch item shares the returned list — no per-item CPU copy and
        no per-item wrap/validate.
        """
        calls = self._spy_from_cpu_indices(monkeypatch)
        model = make_model(DynamicAxisMxq(vocab_size=5, max_width=4), max_batch_size=4)
        attention_mask = torch.tensor(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.long
        )
        self._run_batched(
            model,
            attention_mask=attention_mask,
            logits_to_keep=torch.tensor([0, 2]),
            prefill_chunk_size=3,
        )
        assert len(calls) == 1
        assert calls[0][1] == 3

    def test_lens_mixed_shares_single_cpu_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mixed-length batch + tensor selector: wrap/validate necessarily runs
        per seq_len, but every call must receive the SAME list identity —
        proving the device->host copy was hoisted out of the loop.
        """
        calls = self._spy_from_cpu_indices(monkeypatch)
        model = make_model(DynamicAxisMxq(vocab_size=5, max_width=4), max_batch_size=3)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long
        )
        self._run_batched(
            model,
            attention_mask=attention_mask,
            logits_to_keep=torch.tensor([0, 1]),
            prefill_chunk_size=4,
        )
        assert len(calls) == 3
        shared_ids = {c[0] for c in calls}
        assert len(shared_ids) == 1, (
            "device->host copy must be hoisted: every _output_positions_from_cpu_indices "
            "call must receive the same list object, but got distinct ids per call"
        )
        assert sorted(c[1] for c in calls) == [2, 3, 4]

    def test_int_selector_skips_from_cpu_indices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Int selectors (non-tensor) have no device->host copy to hoist, so
        the CPU-indices entry point must not be involved.
        """
        calls = self._spy_from_cpu_indices(monkeypatch)
        model = make_model(DynamicAxisMxq(vocab_size=5, max_width=4), max_batch_size=2)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
        # int == 0 is not the default fast path (that's int == 1), so this
        # exercises the shared branch with an int selector.
        self._run_batched(
            model, attention_mask=attention_mask, logits_to_keep=0, prefill_chunk_size=3
        )
        assert calls == []


# ---------------------------------------------------------------------------
# Path 3 fallback phase classification for count_npu_time
# ---------------------------------------------------------------------------


class TestBatchPath3PhaseTiming:
    """Path 3 chunk=1 capture calls are decode-shaped work (cursor > 0, cache
    already populated) but the auto-heuristic inside ``_run_batch_infer`` keys
    on ``max_sequence_length`` (batch-wide max, fixed at entry) — so without an
    explicit override every capture would be misfiled under ``prefill_time``
    and inflate the benchmark's prefill accounting.
    """

    def _capture_phase_calls(self, model) -> list[str]:
        """Wrap ``_record_npu_timing`` so we can assert the phase ordering.

        We hook on the instance rather than monkeypatching the class so
        parallel test workers do not race, and so a bound-method wrap sees
        the same ``self`` the production call site would use.
        """
        phases: list[str] = []
        original = model._record_npu_timing

        def spy(phase, elapsed):
            phases.append(phase)
            return original(phase, elapsed)

        model._record_npu_timing = spy
        return phases

    def _run_batched(
        self,
        model,
        *,
        attention_mask: torch.Tensor,
        logits_to_keep,
        hidden_size: int = 4,
        prefill_chunk_size: int = 4,
    ):
        batch, seq_len = attention_mask.shape
        inputs_embeds = torch.arange(batch * seq_len * hidden_size, dtype=torch.float32).reshape(
            batch, seq_len, hidden_size
        )
        cache_position = torch.arange(seq_len)
        return model.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            cache_position=cache_position,
            prefill_chunk_size=prefill_chunk_size,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            count_npu_time=True,
        )

    def test_path3_chunk1_capture_records_decode_phase(self) -> None:
        """The size-1 capture at a kept position must be filed as decode.

        Trace for ``logits_to_keep=[2, 3]`` on two items of length 4 with
        ``prefill_chunk_size=2``:

        - cursor 0, chunk=2 (KV walk 0→2, no capture)              → prefill
        - cursor 2, chunk=1 (capture position 2 for both items)    → decode
        - cursor 3, chunk=1 (capture position 3 for both items)    → decode

        The assertion pins each phase in order; a regression that reverts
        Path 3 to the batch-wide auto-heuristic would flip both chunk=1
        calls back to prefill.
        """
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        model = make_model(mxq, max_batch_size=4)
        phases = self._capture_phase_calls(model)

        self._run_batched(
            model,
            attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
            logits_to_keep=torch.tensor([2, 3]),
            prefill_chunk_size=2,
        )

        assert phases == ["prefill", "decode", "decode"]
        timing = model.get_npu_timing()
        assert timing["prefill_calls"] == 1
        assert timing["decode_calls"] == 2

    def test_path3_bulk_walk_at_cursor_gt_zero_still_records_prefill(self) -> None:
        """Chunk>1 walks (bulk KV extension between captures) stay prefill.

        Only the size-1 capture calls are decode-shaped; a chunk>1 walk at
        cursor > 0 is still populating the KV cache from real input tokens,
        i.e. prefill. Anchors the narrow scope of the fix so a future change
        doesn't over-broaden the override to every cursor > 0 call.

        With ``logits_to_keep=[-1]`` and asymmetric lengths [6, 1], the walk
        goes: chunk=1 @0 (both), chunk=4 @1 (item 0 only, bulk walk), chunk=1
        @5 (item 0 capture). The middle call is chunk=4 at cursor=1 with a
        non-zero cache — a regression that overrides *every* Path 3 call to
        decode would flip its phase.
        """
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        model = make_model(mxq, max_batch_size=4)
        phases = self._capture_phase_calls(model)

        self._run_batched(
            model,
            attention_mask=torch.tensor(
                [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]], dtype=torch.long
            ),
            logits_to_keep=torch.tensor([0, -1]),
            prefill_chunk_size=4,
        )

        assert phases == ["decode", "prefill", "decode"]

    def test_path1_default_keep_phase_uses_auto_heuristic(self) -> None:
        """Path 1 (logits_to_keep=1) never sets phase_override, so its
        classification must still come from the auto-heuristic. All chunks
        run at ``max_sequence_length > 1`` → prefill.
        """
        mxq = StaticLastOnlyMxq(vocab_size=5, max_width=4)
        model = make_model(mxq, max_batch_size=4)
        phases = self._capture_phase_calls(model)

        self._run_batched(
            model,
            attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
            logits_to_keep=1,
            prefill_chunk_size=2,
        )

        assert phases == ["prefill", "prefill"]
        timing = model.get_npu_timing()
        assert timing["decode_calls"] == 0
        assert timing["prefill_calls"] == 2

    def test_path2_dynamic_axis_phase_uses_auto_heuristic(self) -> None:
        """Path 2 (dynamic-axis all-logits) also leaves phase_override=None.
        Every chunk fires with ``max_sequence_length > 1`` → prefill.
        """
        mxq = DynamicAxisMxq(vocab_size=5, max_width=4)
        model = make_model(mxq, max_batch_size=4)
        phases = self._capture_phase_calls(model)

        self._run_batched(
            model,
            attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
            logits_to_keep=0,
            prefill_chunk_size=2,
        )

        assert phases == ["prefill", "prefill"]
        timing = model.get_npu_timing()
        assert timing["decode_calls"] == 0
        assert timing["prefill_calls"] == 2
