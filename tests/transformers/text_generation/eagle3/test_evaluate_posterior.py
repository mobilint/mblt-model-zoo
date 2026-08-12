"""Numerical-parity tests for :func:`evaluate_posterior` greedy-path refactor.

The refactor swaps the order of ``logits[safe_positions]`` fancy indexing and
``torch.argmax`` in the greedy branch to avoid materializing an
``(n_cand, depth-1, vocab)`` intermediate. These tests pin the returned tuple to
match a reference implementation that mirrors the pre-refactor behavior.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
from transformers.generation.logits_process import LogitsProcessorList

from mblt_model_zoo.hf_transformers.utils.eagle3.tree_decoding import (
    evaluate_posterior,
    prepare_logits_processor,
)


def _reference_evaluate_posterior_greedy(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference greedy-path implementation matching the pre-refactor code exactly.

    Args:
        logits: ``(n_tree_nodes, vocab)`` logits tensor.
        candidates: ``(n_cand, depth)`` candidate token tensor.
        retrieve_indices: ``(n_cand, depth)`` tree-position tensor with ``-1`` for padding.

    Returns:
        Tuple ``(best_candidate, accepted_draft_count, sample_p)`` computed the old way.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    path_positions = retrieve_indices[:, :-1].to(logits.device)
    safe_positions = path_positions.clamp_min(0)
    path_logits = logits[safe_positions]
    greedy_tokens = torch.argmax(path_logits, dim=-1)
    candidate_targets = candidates[:, 1:].to(logits.device)
    valid_mask = (path_positions >= 0) & (candidate_targets >= 0)
    posterior_mask = ((candidate_targets == greedy_tokens) & valid_mask).int()
    accepted_draft_counts = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
    accepted_draft_count = accepted_draft_counts.max()
    if accepted_draft_count == 0:
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(accepted_draft_counts).to(torch.long)

    leaf_position = retrieve_indices[best_candidate, accepted_draft_count].to(logits.device)
    if 0 <= int(leaf_position.item()) < logits.shape[0]:
        sample_p = logits[leaf_position]
    else:
        sample_index = torch.clamp(accepted_draft_count, max=path_logits.shape[1] - 1)
        sample_p = path_logits[best_candidate, sample_index]
    return best_candidate, accepted_draft_count, sample_p


def _make_matching_candidates(
    logits: torch.Tensor,
    retrieve_indices: torch.Tensor,
    accept_up_to: int,
) -> torch.Tensor:
    """Build a ``candidates`` tensor whose leading ``accept_up_to`` tokens are greedy matches.

    Args:
        logits: ``(n_tree_nodes, vocab)`` logits tensor.
        retrieve_indices: ``(n_cand, depth)`` tree-position tensor with ``-1`` padding.
        accept_up_to: Number of positions per row to fill with the greedy token; the
            remaining positions get random non-negative token IDs (so acceptance stops
            earlier only for rows that diverge from the greedy trace on their own).

    Returns:
        ``(n_cand, depth)`` integer tensor whose column 0 is a placeholder and columns
        ``1..accept_up_to`` are the greedy tokens for the corresponding tree nodes;
        padded rows retain ``-1`` where ``retrieve_indices == -1``.
    """
    n_cand, depth = retrieve_indices.shape
    vocab = logits.shape[-1]
    greedy_per_node = torch.argmax(logits, dim=-1)
    generator = torch.Generator().manual_seed(1234)
    candidates = torch.randint(0, vocab, (n_cand, depth), generator=generator, dtype=torch.long)
    candidates[:, 0] = 0
    path_positions = retrieve_indices[:, :-1]
    for row in range(n_cand):
        for col in range(min(accept_up_to, depth - 1)):
            pos = int(path_positions[row, col].item())
            if pos < 0:
                candidates[row, col + 1] = -1
            else:
                candidates[row, col + 1] = greedy_per_node[pos]
        for col in range(depth - 1):
            if int(path_positions[row, col].item()) < 0:
                candidates[row, col + 1] = -1
    return candidates


@pytest.mark.parametrize(
    "n_tree_nodes,n_cand,depth,vocab,seed",
    [
        (8, 4, 5, 32, 0),
        (16, 6, 4, 128, 7),
        (24, 10, 6, 256, 13),
    ],
)
def test_greedy_matches_reference_across_shapes(
    n_tree_nodes: int, n_cand: int, depth: int, vocab: int, seed: int
) -> None:
    """New greedy path returns tensors identical to the pre-refactor implementation."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(n_tree_nodes, vocab, generator=generator, dtype=torch.float32)
    retrieve_indices = torch.randint(0, n_tree_nodes, (n_cand, depth), generator=generator, dtype=torch.long)
    candidates = torch.randint(0, vocab, (n_cand, depth), generator=generator, dtype=torch.long)
    candidates[:, 0] = 0

    ref_best, ref_count, ref_sample_p = _reference_evaluate_posterior_greedy(logits, candidates, retrieve_indices)
    new_best, new_count, new_sample_p, new_sampled = evaluate_posterior(logits, candidates, None, retrieve_indices)

    assert new_sampled is None
    assert torch.equal(new_best, ref_best)
    assert torch.equal(new_count, ref_count)
    assert torch.equal(new_sample_p, ref_sample_p)


def test_greedy_full_accept_hits_leaf_sample_p() -> None:
    """When the full path is accepted with a valid leaf node, sample_p reads logits[leaf]."""
    torch.manual_seed(42)
    n_tree_nodes, depth, vocab = 12, 4, 64
    logits = torch.randn(n_tree_nodes, vocab)
    # Build a small tree where every retrieve_indices entry is a valid node.
    retrieve_indices = torch.tensor(
        [
            [0, 1, 4, 8],
            [0, 2, 5, 9],
            [0, 3, 6, 10],
        ],
        dtype=torch.long,
    )
    candidates = _make_matching_candidates(logits, retrieve_indices, accept_up_to=depth - 1)

    ref_best, ref_count, ref_sample_p = _reference_evaluate_posterior_greedy(logits, candidates, retrieve_indices)
    new_best, new_count, new_sample_p, new_sampled = evaluate_posterior(logits, candidates, None, retrieve_indices)

    assert new_sampled is None
    assert torch.equal(new_best, ref_best)
    assert torch.equal(new_count, ref_count)
    assert torch.equal(new_sample_p, ref_sample_p)
    # With a full accept and valid leaf, sample_p must be the logits row at the winning leaf.
    leaf = int(retrieve_indices[new_best, new_count].item())
    assert 0 <= leaf < n_tree_nodes
    assert torch.equal(new_sample_p, logits[leaf])


def test_greedy_fallback_when_leaf_position_is_padding() -> None:
    """When the winning path's leaf slot is ``-1`` padding, both impls take the fallback branch."""
    torch.manual_seed(2026)
    n_tree_nodes, vocab = 6, 32
    logits = torch.randn(n_tree_nodes, vocab)
    # Single path with retrieve_indices ending in a padding sentinel so accepted_draft_count
    # lands on a ``-1`` slot and forces the fallback branch.
    retrieve_indices = torch.tensor([[0, 3, -1, -1]], dtype=torch.long)
    greedy_per_node = torch.argmax(logits, dim=-1)
    # candidates[i, j+1] must equal greedy at retrieve_indices[i, j] to be accepted.
    candidates = torch.tensor(
        [[0, int(greedy_per_node[0].item()), int(greedy_per_node[3].item()), -1]],
        dtype=torch.long,
    )

    ref_best, ref_count, ref_sample_p = _reference_evaluate_posterior_greedy(logits, candidates, retrieve_indices)
    new_best, new_count, new_sample_p, new_sampled = evaluate_posterior(logits, candidates, None, retrieve_indices)

    assert new_sampled is None
    assert torch.equal(new_best, ref_best)
    assert torch.equal(new_count, ref_count)
    assert torch.equal(new_sample_p, ref_sample_p)
    assert int(new_count.item()) == 2  # sanity: two valid tokens accepted
    leaf_position = int(retrieve_indices[new_best, new_count].item())
    assert leaf_position < 0 or leaf_position >= n_tree_nodes  # fallback branch actually hit
    # In the fallback, sample_p must equal logits at the clamped, safe-positioned node.
    sample_index = int(torch.clamp(new_count, max=retrieve_indices.shape[1] - 2).item())
    safe_positions = retrieve_indices[:, :-1].clamp_min(0)
    fallback_node = int(safe_positions[new_best, sample_index].item())
    assert torch.equal(new_sample_p, logits[fallback_node])


def test_greedy_zero_accept_returns_first_candidate() -> None:
    """When no draft token matches the greedy prediction, best_candidate is 0 and count is 0."""
    torch.manual_seed(11)
    n_tree_nodes, n_cand, depth, vocab = 8, 4, 4, 64
    logits = torch.randn(n_tree_nodes, vocab)
    retrieve_indices = torch.randint(0, n_tree_nodes, (n_cand, depth), dtype=torch.long)
    greedy_per_node = torch.argmax(logits, dim=-1)
    # Build candidates whose column-1 token is guaranteed to differ from greedy_per_node.
    candidates = torch.zeros((n_cand, depth), dtype=torch.long)
    for row in range(n_cand):
        node = int(retrieve_indices[row, 0].item())
        candidates[row, 1] = (int(greedy_per_node[node].item()) + 1) % vocab

    ref_best, ref_count, ref_sample_p = _reference_evaluate_posterior_greedy(logits, candidates, retrieve_indices)
    new_best, new_count, new_sample_p, new_sampled = evaluate_posterior(logits, candidates, None, retrieve_indices)

    assert new_sampled is None
    assert int(new_count.item()) == 0
    assert int(new_best.item()) == 0
    assert torch.equal(new_best, ref_best)
    assert torch.equal(new_count, ref_count)
    assert torch.equal(new_sample_p, ref_sample_p)


def test_greedy_accepts_1d_logits() -> None:
    """A 1D logits input is unsqueezed to 2D and still returns matching outputs."""
    torch.manual_seed(3)
    vocab = 32
    logits_1d = torch.randn(vocab)
    retrieve_indices = torch.tensor([[0, -1]], dtype=torch.long)
    candidates = torch.tensor([[0, -1]], dtype=torch.long)

    ref_best, ref_count, ref_sample_p = _reference_evaluate_posterior_greedy(logits_1d, candidates, retrieve_indices)
    new_best, new_count, new_sample_p, new_sampled = evaluate_posterior(logits_1d, candidates, None, retrieve_indices)

    assert new_sampled is None
    assert torch.equal(new_best, ref_best)
    assert torch.equal(new_count, ref_count)
    assert torch.equal(new_sample_p, ref_sample_p)


def test_sampling_path_unaffected_by_refactor() -> None:
    """The refactor lives strictly in the greedy branch; sampling still returns 4-tuples with sampled_indices."""
    torch.manual_seed(19)
    n_tree_nodes, n_cand, depth, vocab = 8, 3, 3, 64
    logits = torch.randn(n_tree_nodes, vocab)
    retrieve_indices = torch.tensor(
        [
            [0, 1, 4],
            [0, 2, 5],
            [0, 3, 6],
        ],
        dtype=torch.long,
    )
    greedy_per_node = torch.argmax(logits, dim=-1)
    candidates = torch.zeros((n_cand, depth), dtype=torch.long)
    for row in range(n_cand):
        node = int(retrieve_indices[row, 0].item())
        candidates[row, 1] = greedy_per_node[node]

    processor: Optional[LogitsProcessorList] = prepare_logits_processor(temperature=0.7)
    assert processor is not None

    torch.manual_seed(19)
    best, count, sample_p, sampled_indices = evaluate_posterior(logits, candidates, processor, retrieve_indices)
    assert isinstance(best, torch.Tensor) and best.dtype in (torch.long, torch.int64)
    assert isinstance(count, torch.Tensor) and count.dtype in (torch.long, torch.int64)
    assert isinstance(sample_p, torch.Tensor)
    # The sampling branch always exercises softmax_topk_cpu_torch and therefore returns indices.
    assert sampled_indices is not None
    assert sampled_indices.dtype == torch.long
