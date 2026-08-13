"""Unit tests for EAGLE-3 acceptance accounting under EOS and budget truncation.

The decode loop must record acceptance from the actual emitted-length delta so
tokens dropped by ``update_inference_inputs`` (EOS in-candidate, remaining
budget) never inflate ``acceptance_tokens_sum`` or downstream ratios.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.generation.stopping_criteria import StoppingCriteriaList

from mblt_model_zoo.hf_transformers.utils.generation_utils import MobilintEagle3GenerationMixin


class _FakeCache:
    """Minimal ``MobilintEagle3Cache`` stand-in for the decode loop."""

    def __init__(self) -> None:
        self.accept_tokens: torch.Tensor | None = None

    def update_base_seen_tokens(self, _length: int) -> None:  # pragma: no cover - exercised only on stop
        pass


def _make_fake_self() -> SimpleNamespace:
    """Build a ``self`` stub sufficient for ``_run_eagle3_decode_loop``."""
    fake = SimpleNamespace()
    fake._eagle3_stopping_scores_adapter = MobilintEagle3GenerationMixin._eagle3_stopping_scores_adapter
    return fake


def _patch_decoding(
    monkeypatch: pytest.MonkeyPatch,
    *,
    emit_lengths: list[int],
    accepted_draft_counts: list[int],
    should_stops: list[bool],
) -> None:
    """Patch the four decoding-module entries used by the loop.

    ``update_inference_inputs`` is stubbed so ``generated.shape[1]`` grows by
    ``emit_lengths[i]`` on iteration ``i``. That delta is exactly what the
    decode loop reads to derive accepted drafts, so any mismatch here is what
    the guarded assertion detects.
    """
    from mblt_model_zoo.hf_transformers.utils.eagle3 import decoding as decoding_module

    monkeypatch.setattr(
        decoding_module,
        "initialize_tree",
        lambda *_args, **_kwargs: (
            torch.tensor([[10, 11, 12, 13]], dtype=torch.long),  # draft_tokens (1, 4)
            torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),  # retrieve_indices (1, 5)
            None,
            torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
            None,
        ),
    )

    monkeypatch.setattr(
        decoding_module,
        "tree_decoding",
        lambda *_args, **_kwargs: (
            torch.zeros((1, 5, 16), dtype=torch.float32),  # logits shaped (batch, vocab_slots, vocab)
            torch.zeros((1, 5, 1), dtype=torch.float32),  # hidden_state_new
        ),
    )

    step_index = {"i": 0}

    def _fake_evaluate_posterior(*_args, **_kwargs):
        i = step_index["i"]
        return (
            torch.tensor(0, dtype=torch.long),  # best_candidate
            accepted_draft_counts[i],
            torch.zeros((1, 16), dtype=torch.float32),  # sample_p
            torch.zeros((1,), dtype=torch.long),  # sampled_indices
        )

    def _fake_update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accepted_draft_count,
        retrieve_indices,
        logits_processor,
        new_token_count,
        model,
        cache,
        hidden_state_new,
        sample_p,
        sampled_indices,
        *,
        remaining_tokens=None,
        eos_token_id=None,
        count_npu_time=False,
    ):
        i = step_index["i"]
        emit = emit_lengths[i]
        step_index["i"] = i + 1
        if emit > 0:
            appended = torch.full((1, emit), 99, dtype=input_ids.dtype, device=input_ids.device)
            new_generated = torch.cat([input_ids, appended], dim=-1)
        else:
            new_generated = input_ids
        stop = should_stops[i]
        # Terminating iterations reset draft/tree state to empty (matches the
        # real ``update_inference_inputs`` short-circuit). Continuing
        # iterations must ship a valid draft-tree state so the next round can
        # rebuild ``candidates`` with the same width. Reusing the same shapes
        # ``initialize_tree`` produces keeps ``candidate_draft_tokens`` stable
        # across iterations.
        if stop:
            next_draft_tokens = torch.empty(0, dtype=torch.long)
            next_retrieve_indices = torch.empty(0, dtype=torch.long)
        else:
            next_draft_tokens = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
            next_retrieve_indices = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        return (
            new_generated,
            next_draft_tokens,
            next_retrieve_indices,
            torch.empty(0),
            torch.empty(0),
            new_token_count + emit,
            stop,
        )

    monkeypatch.setattr(decoding_module, "evaluate_posterior", _fake_evaluate_posterior)
    monkeypatch.setattr(decoding_module, "update_inference_inputs", _fake_update_inference_inputs)


def _run_loop(monkeypatch: pytest.MonkeyPatch, *, emit_lengths, accepted_draft_counts, should_stops, max_tokens: int):
    _patch_decoding(
        monkeypatch,
        emit_lengths=emit_lengths,
        accepted_draft_counts=accepted_draft_counts,
        should_stops=should_stops,
    )
    fake_self = _make_fake_self()
    generated = torch.tensor([[1, 2]], dtype=torch.long)
    result = MobilintEagle3GenerationMixin._run_eagle3_decode_loop(
        fake_self,
        generated=generated,
        cache=_FakeCache(),
        logits_processor=None,
        max_tokens=max_tokens,
        eos_token_id=None,
        stopping_criteria_list=StoppingCriteriaList(),
        streamer=None,
        count_npu_time=False,
    )
    return result, fake_self._eagle3_acceptance_stats


def test_eagle3_decode_loop_records_full_iteration_without_truncation(monkeypatch):
    """Normal iter: emit 4 tokens (3 drafts + 1 root); tally the 3 accepted drafts."""
    _, stats = _run_loop(
        monkeypatch,
        emit_lengths=[4],
        accepted_draft_counts=[3],
        should_stops=[True],
        max_tokens=64,
    )

    assert stats["steps"] == 1
    assert stats["accepted_tokens_sum"] == 3
    assert stats["accepted_tokens_avg"] == pytest.approx(3.0)
    # candidate_draft_tokens == retrieve_indices.shape[-1] - 1 == 4.
    assert stats["acceptance_ratio"] == pytest.approx(3.0 / 4.0)


def test_eagle3_decode_loop_records_eos_truncated_iteration(monkeypatch):
    """EOS shrinks emit to 2 even though posterior accepted 3 drafts."""
    _, stats = _run_loop(
        monkeypatch,
        emit_lengths=[2],
        accepted_draft_counts=[3],
        should_stops=[True],
        max_tokens=64,
    )

    assert stats["steps"] == 1
    assert stats["accepted_tokens_sum"] == 1  # emit=2 → drafts = emit-1 = 1
    assert stats["accepted_tokens_avg"] == pytest.approx(1.0)
    assert stats["acceptance_ratio"] == pytest.approx(1.0 / 4.0)


def test_eagle3_decode_loop_records_budget_truncated_iteration(monkeypatch):
    """Remaining-token budget shrinks emit to 2 even though posterior accepted 4 drafts."""
    _, stats = _run_loop(
        monkeypatch,
        emit_lengths=[2],
        accepted_draft_counts=[4],
        should_stops=[True],
        max_tokens=2,
    )

    assert stats["steps"] == 1
    assert stats["accepted_tokens_sum"] == 1
    assert stats["accepted_tokens_avg"] == pytest.approx(1.0)
    assert stats["acceptance_ratio"] == pytest.approx(1.0 / 4.0)


def test_eagle3_decode_loop_handles_zero_emission_iteration(monkeypatch):
    """Empty emission (no accepted candidate survived truncation): step counts, tokens = 0."""
    _, stats = _run_loop(
        monkeypatch,
        emit_lengths=[0],
        accepted_draft_counts=[2],
        should_stops=[True],
        max_tokens=64,
    )

    assert stats["steps"] == 1
    assert stats["accepted_tokens_sum"] == 0
    assert stats["accepted_tokens_avg"] == pytest.approx(0.0)
    assert stats["acceptance_ratio"] == pytest.approx(0.0)


def test_eagle3_decode_loop_averages_across_normal_then_truncated(monkeypatch):
    """Mixed run: full iter then a budget-truncated final iter averages correctly."""
    _, stats = _run_loop(
        monkeypatch,
        emit_lengths=[4, 2],  # iter 0 emits 4 (3 drafts), iter 1 emits 2 (1 draft) — budget hit
        accepted_draft_counts=[3, 4],
        should_stops=[False, True],
        max_tokens=6,
    )

    assert stats["steps"] == 2
    assert stats["accepted_tokens_sum"] == 4  # 3 + 1
    assert stats["accepted_tokens_avg"] == pytest.approx(2.0)
    assert stats["acceptance_ratio"] == pytest.approx((3.0 / 4.0 + 1.0 / 4.0) / 2.0)
