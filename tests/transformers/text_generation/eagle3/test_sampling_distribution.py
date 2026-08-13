"""HF-parity tests for the EAGLE-3 root/next-token sampling helper.

These tests pin :func:`_sample_next_token_from_processor` (the sampling counterpart to
:func:`softmax_topk_cpu_torch`) to the reference Hugging Face contract
``LogitsProcessorList`` + ``softmax`` + ``multinomial``. The P1 review comment on PR #107
observed that passing the top-N slice from ``softmax_topk_cpu_torch`` straight to
``torch.multinomial`` silently renormalized over 10 tokens and gave every other token
zero probability, which broke temperature-only sampling and ``top_k`` values above 10.
Root/next-token sampling now goes through the full-vocab helper instead.
"""

from __future__ import annotations

import pytest
import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from mblt_model_zoo.hf_transformers.utils.eagle3.tree_decoding import (
    _sample_next_token_from_processor,
    prepare_logits_processor,
)


def _make_temperature_only_processor(temperature: float = 1.0) -> LogitsProcessorList:
    """Build a bare-temperature processor list without going through the ``!=1.0`` guard.

    :func:`prepare_logits_processor` intentionally returns ``None`` when the only warper
    would be ``TemperatureLogitsWarper(1.0)`` (a no-op). These tests need the actual list
    in hand to feed :func:`_sample_next_token_from_processor`, so wrap the warper
    unconditionally.
    """
    return LogitsProcessorList([TemperatureLogitsWarper(temperature)])


def _hf_reference_sample_distribution(
    logits: torch.Tensor,
    processor: LogitsProcessorList,
) -> torch.Tensor:
    """Return the HF-processed sampling distribution over the full vocabulary."""
    x = logits.float()
    if x.ndim == 1:
        expanded = processor(None, x.unsqueeze(0))[0]
    else:
        expanded = processor(None, x)
    return torch.softmax(expanded, dim=-1)


def _empirical_distribution(samples: torch.Tensor, vocab: int) -> torch.Tensor:
    """Return an empirical ``(vocab,)`` distribution from a 1D long tensor of samples."""
    counts = torch.bincount(samples.view(-1), minlength=vocab).float()
    return counts / counts.sum()


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Return the KL divergence ``D(p || q)`` guarded against zeros."""
    mask = p > 0
    p_masked = p[mask]
    q_masked = q[mask].clamp_min(1e-12)
    return float((p_masked * (torch.log(p_masked) - torch.log(q_masked))).sum().item())


@pytest.fixture
def sample_logits() -> torch.Tensor:
    """Return a 1D logits vector small enough that KL noise stays tight at moderate sample counts."""
    generator = torch.Generator().manual_seed(2026)
    return torch.randn(256, generator=generator, dtype=torch.float32)


def test_helper_returns_long_shape_matches_1d_input(sample_logits) -> None:
    """1D input returns shape ``(1,)`` and dtype ``torch.long`` (compatible with argmax path)."""
    processor = _make_temperature_only_processor(1.0)
    torch.manual_seed(0)
    token = _sample_next_token_from_processor(sample_logits, processor)
    assert token.dtype == torch.long
    assert token.shape == (1,)
    assert 0 <= int(token.item()) < sample_logits.numel()


def test_helper_returns_shape_batch_by_one_for_2d_input() -> None:
    """2D input returns shape ``(batch, 1)`` (matching :func:`initialize_tree`)."""
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 512, generator=generator, dtype=torch.float32)
    processor = _make_temperature_only_processor(1.0)
    torch.manual_seed(0)
    token = _sample_next_token_from_processor(logits, processor)
    assert token.dtype == torch.long
    assert token.shape == (3, 1)


def test_temperature_only_matches_hf_full_vocab(sample_logits) -> None:
    """Temperature-only sampling matches HF full-vocab softmax + multinomial in distribution.

    Under the old code path, the token pool was clipped to the top-10 indices, so tokens
    outside the top-10 had zero probability. HF's contract has probability spread across
    the whole vocab, so out-of-top-10 tokens must appear with meaningful frequency.
    """
    processor = _make_temperature_only_processor(1.0)
    reference_processor = _make_temperature_only_processor(1.0)

    reference_probs = _hf_reference_sample_distribution(sample_logits, reference_processor)

    torch.manual_seed(0)
    num_samples = 40000
    tokens = torch.stack(
        [_sample_next_token_from_processor(sample_logits, processor).squeeze(0) for _ in range(num_samples)]
    )
    empirical = _empirical_distribution(tokens, sample_logits.numel())

    # The old top-10 slice contract confined the empirical support to at most 10 tokens.
    # The full-vocab contract spreads mass so that far more tokens appear at temp 1.0.
    assert int((empirical > 0).sum().item()) > 40

    # KL(p_ref || p_emp) should be small when sampling from the same distribution. The
    # threshold accommodates finite-sample noise at ``num_samples``; it is orders of
    # magnitude tighter than a top-10 renormalized empirical would produce.
    kl = _kl_divergence(reference_probs, empirical)
    assert kl < 0.1


def test_top_k_50_lets_tokens_outside_top_10_get_sampled() -> None:
    """``top_k=50`` (above the candidate-matching return slice of 10) permits out-of-top-10 samples.

    The old top-N sampling would have masked every token beyond index 10 to zero
    probability. With full-vocab sampling, HF's top-50 nucleus is honored end-to-end.
    """
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(1024, generator=generator, dtype=torch.float32)
    processor = prepare_logits_processor(temperature=1.0, top_k=50)
    assert processor is not None
    reference_processor = prepare_logits_processor(temperature=1.0, top_k=50)
    assert reference_processor is not None

    reference_probs = _hf_reference_sample_distribution(logits, reference_processor)
    # Sanity: at least 50 tokens carry mass in the reference distribution.
    assert int((reference_probs > 0).sum().item()) == 50

    torch.manual_seed(0)
    num_samples = 40000
    tokens = torch.stack([_sample_next_token_from_processor(logits, processor).squeeze(0) for _ in range(num_samples)])
    empirical = _empirical_distribution(tokens, logits.numel())

    top10_indices = torch.topk(logits, 10).indices
    out_of_top10_mass = float(empirical.sum().item()) - float(empirical[top10_indices].sum().item())
    # The 40 tokens between top-11 and top-50 carry a substantial share of mass in a normal-
    # distributed vocabulary; require at least 5% to guard against the old top-10
    # renormalization regression while tolerating sampling noise.
    assert out_of_top10_mass > 0.05

    kl = _kl_divergence(reference_probs, empirical)
    assert kl < 0.1


def test_top_k_5_zeros_out_of_nucleus_tokens() -> None:
    """``top_k=5`` (below the return slice) hard-masks tokens 6..end via HF processor semantics."""
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(1024, generator=generator, dtype=torch.float32)
    processor = prepare_logits_processor(temperature=1.0, top_k=5)
    assert processor is not None

    torch.manual_seed(0)
    num_samples = 5000
    tokens = torch.stack([_sample_next_token_from_processor(logits, processor).squeeze(0) for _ in range(num_samples)])
    top5_indices = set(int(idx) for idx in torch.topk(logits, 5).indices.tolist())
    sampled_tokens = set(int(t.item()) for t in tokens)
    # Every sampled token must be within HF's top-5 nucleus (no leakage from the return slice).
    assert sampled_tokens.issubset(top5_indices)


def test_top_p_only_uses_full_vocab_nucleus() -> None:
    """A bare ``top_p`` warper must select the nucleus from the full-vocab distribution."""
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(512, generator=generator, dtype=torch.float32)
    processor = prepare_logits_processor(temperature=1.0, top_p=0.9)
    assert processor is not None
    reference_processor = prepare_logits_processor(temperature=1.0, top_p=0.9)
    assert reference_processor is not None

    reference_probs = _hf_reference_sample_distribution(logits, reference_processor)
    nucleus_indices = set(int(i) for i in (reference_probs > 0).nonzero(as_tuple=True)[0].tolist())

    torch.manual_seed(0)
    num_samples = 5000
    tokens = torch.stack([_sample_next_token_from_processor(logits, processor).squeeze(0) for _ in range(num_samples)])
    sampled_tokens = set(int(t.item()) for t in tokens)
    assert sampled_tokens.issubset(nucleus_indices)

    empirical = _empirical_distribution(tokens, logits.numel())
    kl = _kl_divergence(reference_probs, empirical)
    assert kl < 0.1


def test_helper_processor_order_matches_prepared_list() -> None:
    """The helper honors the HF ``Temperature → TopK → TopP`` warper order established by
    :func:`prepare_logits_processor` (regression guard against reordering)."""
    processor = prepare_logits_processor(temperature=0.7, top_k=50, top_p=0.9)
    assert processor is not None
    warper_types = [type(w) for w in processor]
    assert warper_types.index(TemperatureLogitsWarper) < warper_types.index(TopKLogitsWarper)
    assert warper_types.index(TopKLogitsWarper) < warper_types.index(TopPLogitsWarper)
