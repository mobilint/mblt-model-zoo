"""Unit tests for the ``softmax_topk_cpu_torch`` mode dispatch and HF nucleus contract."""

from __future__ import annotations

import importlib

import pytest
import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from mblt_model_zoo.hf_transformers.utils.eagle3 import tree_decoding as tree_decoding_module
from mblt_model_zoo.hf_transformers.utils.eagle3.tree_decoding import (
    _VALID_SOFTMAX_TOPK_MODES,
    _extract_top_k_from_processor,
    prepare_logits_processor,
    set_softmax_topk_mode,
    softmax_topk_cpu_torch,
)


@pytest.fixture
def restore_softmax_topk_mode():
    """Restore the module-level mode after each test."""
    original_mode = tree_decoding_module.SOFTMAX_TOPK_MODE
    original_logged = tree_decoding_module._last_logged_softmax_topk_mode
    yield
    tree_decoding_module.SOFTMAX_TOPK_MODE = original_mode
    tree_decoding_module._last_logged_softmax_topk_mode = original_logged


@pytest.fixture
def sample_logits() -> torch.Tensor:
    """Return a deterministic 1D logits tensor over a moderately large vocabulary."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(1024, generator=generator, dtype=torch.float32)


def _hf_full_vocab_reference(
    logits: torch.Tensor,
    processor: LogitsProcessorList | None,
    max_return_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference full-vocab implementation used to pin the HF contract."""
    x = logits.float()
    if x.ndim == 1:
        expanded = x.unsqueeze(0)
    else:
        expanded = x
    if processor is not None:
        expanded = processor(None, expanded)
    processed = expanded[0] if x.ndim == 1 else expanded
    return_k = min(int(max_return_k), processed.shape[-1])
    max_val = processed.max(dim=-1, keepdim=True).values
    denom = torch.exp(processed - max_val).sum(dim=-1, keepdim=True)
    topk_vals, topk_idx = torch.topk(processed, return_k, dim=-1, largest=True, sorted=True)
    probs = torch.exp(topk_vals - max_val) / denom
    return probs, topk_idx


def test_default_mode_is_auto():
    """The default mode is the new auto dispatch path."""
    reloaded = importlib.reload(tree_decoding_module)
    try:
        assert reloaded.SOFTMAX_TOPK_MODE == "auto"
    finally:
        importlib.reload(tree_decoding_module)


def test_env_var_selects_full_mode(monkeypatch):
    """A ``full`` env var value forces the full-vocab path at import time."""
    monkeypatch.setenv("MBLT_EAGLE3_SOFTMAX_TOPK_MODE", "full")
    reloaded = importlib.reload(tree_decoding_module)
    try:
        assert reloaded.SOFTMAX_TOPK_MODE == "full"
    finally:
        monkeypatch.delenv("MBLT_EAGLE3_SOFTMAX_TOPK_MODE", raising=False)
        importlib.reload(tree_decoding_module)


def test_env_var_selects_deprecated_sliced_mode(monkeypatch):
    """A ``sliced`` env var value keeps the legacy renormalized-slice path (with a warning)."""
    monkeypatch.setenv("MBLT_EAGLE3_SOFTMAX_TOPK_MODE", "sliced")
    reloaded = importlib.reload(tree_decoding_module)
    try:
        assert reloaded.SOFTMAX_TOPK_MODE == "sliced"
    finally:
        monkeypatch.delenv("MBLT_EAGLE3_SOFTMAX_TOPK_MODE", raising=False)
        importlib.reload(tree_decoding_module)


def test_env_var_invalid_falls_back_to_auto(monkeypatch):
    """An unrecognized env var value falls back to the ``auto`` default."""
    monkeypatch.setenv("MBLT_EAGLE3_SOFTMAX_TOPK_MODE", "not-a-mode")
    reloaded = importlib.reload(tree_decoding_module)
    try:
        assert reloaded.SOFTMAX_TOPK_MODE == "auto"
    finally:
        monkeypatch.delenv("MBLT_EAGLE3_SOFTMAX_TOPK_MODE", raising=False)
        importlib.reload(tree_decoding_module)


def test_set_softmax_topk_mode_rejects_unknown(restore_softmax_topk_mode):
    """The programmatic setter rejects unsupported modes."""
    with pytest.raises(ValueError):
        set_softmax_topk_mode("bogus")  # type: ignore[arg-type]


def test_set_softmax_topk_mode_warns_on_sliced(restore_softmax_topk_mode, caplog):
    """Selecting the legacy ``sliced`` mode emits a deprecation warning."""
    with caplog.at_level("WARNING"):
        set_softmax_topk_mode("sliced")
    assert any("deprecated" in record.message.lower() for record in caplog.records)


def test_prepare_logits_processor_orders_topk_before_topp():
    """Warpers must be appended in HF ``_get_logits_warper`` order (TopK before TopP)."""
    processor = prepare_logits_processor(temperature=0.7, top_k=50, top_p=0.9)
    assert processor is not None
    warper_types = [type(warper) for warper in processor]
    assert TemperatureLogitsWarper in warper_types
    assert TopKLogitsWarper in warper_types
    assert TopPLogitsWarper in warper_types
    assert warper_types.index(TopKLogitsWarper) < warper_types.index(TopPLogitsWarper)


def test_extract_top_k_from_processor_returns_declared_value():
    """Helper returns the declared ``top_k`` from the first ``TopKLogitsWarper`` in the list."""
    processor = prepare_logits_processor(temperature=1.0, top_k=17)
    assert processor is not None
    assert _extract_top_k_from_processor(processor) == 17


def test_extract_top_k_from_processor_returns_none_when_absent():
    """Helper returns ``None`` when the processor list has no ``TopKLogitsWarper``."""
    processor = prepare_logits_processor(temperature=0.7, top_p=0.9)
    assert processor is not None
    assert _extract_top_k_from_processor(processor) is None


@pytest.mark.parametrize("mode", _VALID_SOFTMAX_TOPK_MODES)
def test_all_modes_return_same_topk_indices_no_processor(mode, sample_logits, restore_softmax_topk_mode):
    """All modes must select the same top-k raw indices when no processor is provided."""
    set_softmax_topk_mode("full")
    _, full_idx = softmax_topk_cpu_torch(sample_logits, logits_processor=None)
    set_softmax_topk_mode(mode)
    _, mode_idx = softmax_topk_cpu_torch(sample_logits, logits_processor=None)
    assert torch.equal(full_idx, mode_idx)


def test_auto_without_processor_uses_full_vocab_denominator(sample_logits, restore_softmax_topk_mode):
    """``auto`` with no processor takes the full-vocab path — partial mass on the returned slice."""
    set_softmax_topk_mode("auto")
    probs, _ = softmax_topk_cpu_torch(sample_logits, logits_processor=None)
    total = probs.sum().item()
    assert total <= 1.0
    assert total < 1.0 - 1e-6  # diffuse random logits leave real mass outside the top-10


def test_full_probs_sum_below_one(sample_logits, restore_softmax_topk_mode):
    """The ``full`` mode returns partial softmax mass (sum strictly below 1.0)."""
    set_softmax_topk_mode("full")
    probs, _ = softmax_topk_cpu_torch(sample_logits, logits_processor=None)
    total = probs.sum().item()
    assert total <= 1.0
    assert total < 1.0 - 1e-6


def test_legacy_sliced_probs_sum_to_one(sample_logits, restore_softmax_topk_mode):
    """The deprecated ``sliced`` mode still renormalizes over the returned slice."""
    set_softmax_topk_mode("sliced")
    probs, _ = softmax_topk_cpu_torch(sample_logits, logits_processor=None)
    assert probs.shape == (10,)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_auto_with_topk_matches_hf_full_vocab(sample_logits, restore_softmax_topk_mode):
    """Auto path with a TopK warper matches HF full-vocab softmax on the returned slice."""
    processor = prepare_logits_processor(temperature=0.7, top_k=50)
    assert processor is not None
    set_softmax_topk_mode("auto")
    auto_probs, auto_idx = softmax_topk_cpu_torch(sample_logits, logits_processor=processor)

    # Rebuild an independent processor so HF reference does not consume in-place state.
    reference_processor = prepare_logits_processor(temperature=0.7, top_k=50)
    ref_probs, ref_idx = _hf_full_vocab_reference(sample_logits, reference_processor, 10)
    assert torch.equal(auto_idx, ref_idx)
    assert torch.allclose(auto_probs, ref_probs, atol=1e-6)


def test_auto_with_topk_topp_matches_hf_nucleus(sample_logits, restore_softmax_topk_mode):
    """Auto path with TopK + TopP matches HF full-vocab nucleus decision on the returned slice."""
    processor = prepare_logits_processor(temperature=0.8, top_k=64, top_p=0.9)
    assert processor is not None
    set_softmax_topk_mode("auto")
    auto_probs, auto_idx = softmax_topk_cpu_torch(sample_logits, logits_processor=processor)

    reference_processor = prepare_logits_processor(temperature=0.8, top_k=64, top_p=0.9)
    ref_probs, ref_idx = _hf_full_vocab_reference(sample_logits, reference_processor, 10)
    assert torch.equal(auto_idx, ref_idx)
    assert torch.allclose(auto_probs, ref_probs, atol=1e-6)


def test_auto_with_tight_topp_zeros_out_of_nucleus_return_slice(restore_softmax_topk_mode):
    """A tight TopP masks return-slice entries outside the nucleus to exactly zero."""
    generator = torch.Generator().manual_seed(11)
    # Construct logits with a very sharp peak so the nucleus is a small handful of tokens.
    logits = torch.full((256,), -10.0, dtype=torch.float32)
    peak_idx = torch.randperm(256, generator=generator)[:3]
    logits[peak_idx] = torch.tensor([5.0, 4.5, 4.0])
    processor = prepare_logits_processor(temperature=1.0, top_k=32, top_p=0.5)
    assert processor is not None
    set_softmax_topk_mode("auto")
    probs, _ = softmax_topk_cpu_torch(logits, max_return_k=10, logits_processor=processor)
    assert probs.shape == (10,)
    # Nucleus is a single token (the sharpest peak already exceeds top_p=0.5); the rest are zero.
    assert (probs == 0).sum().item() >= 9
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_auto_topp_only_uses_full_vocab_nucleus(sample_logits, restore_softmax_topk_mode):
    """Auto path with only TopP must fall back to full-vocab so nucleus is determined properly."""
    processor = prepare_logits_processor(temperature=0.8, top_p=0.5)
    assert processor is not None
    set_softmax_topk_mode("auto")
    auto_probs, auto_idx = softmax_topk_cpu_torch(sample_logits, logits_processor=processor)

    reference_processor = prepare_logits_processor(temperature=0.8, top_p=0.5)
    ref_probs, ref_idx = _hf_full_vocab_reference(sample_logits, reference_processor, 10)
    assert torch.equal(auto_idx, ref_idx)
    assert torch.allclose(auto_probs, ref_probs, atol=1e-6)


def test_auto_topp_only_diverges_from_legacy_sliced(sample_logits, restore_softmax_topk_mode):
    """TopP-only: the deprecated slice-first path decides a different nucleus than HF."""
    processor = prepare_logits_processor(temperature=0.8, top_p=0.5)
    assert processor is not None

    set_softmax_topk_mode("auto")
    auto_probs, _ = softmax_topk_cpu_torch(sample_logits, logits_processor=processor)

    reference_processor = prepare_logits_processor(temperature=0.8, top_p=0.5)
    set_softmax_topk_mode("sliced")
    sliced_probs, _ = softmax_topk_cpu_torch(sample_logits, logits_processor=reference_processor)

    # The two paths are mathematically different when TopP has no TopK companion; this test
    # documents the divergence that the P1 review flagged, and pins auto to the HF answer.
    assert not torch.allclose(auto_probs, sliced_probs, atol=1e-6)


def test_auto_with_topk_smaller_than_max_return_k(restore_softmax_topk_mode):
    """When declared ``top_k`` is smaller than the return slice, out-of-K probs are exactly zero."""
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(256, generator=generator, dtype=torch.float32)
    processor = prepare_logits_processor(temperature=1.0, top_k=5)
    assert processor is not None
    set_softmax_topk_mode("auto")
    probs, _ = softmax_topk_cpu_torch(logits, max_return_k=10, logits_processor=processor)
    assert probs.shape == (10,)
    assert (probs[:5] > 0).all()
    assert torch.equal(probs[5:], torch.zeros(5, dtype=probs.dtype))
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_auto_mode_supports_2d_logits(restore_softmax_topk_mode):
    """The 2D logits path returns per-row top-``max_return_k`` slices under auto mode."""
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 128, generator=generator, dtype=torch.float32)
    processor = prepare_logits_processor(temperature=1.0, top_k=10)
    assert processor is not None
    set_softmax_topk_mode("auto")
    probs, indices = softmax_topk_cpu_torch(logits, logits_processor=processor)
    assert probs.shape == (3, 10)
    assert indices.shape == (3, 10)
    # When top_k == max_return_k, the return slice is the full nucleus and probs sum to 1.
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_softmax_topk_default_max_return_k_is_ten(sample_logits, restore_softmax_topk_mode):
    """Callers that omit ``max_return_k`` still get a top-10 return slice."""
    set_softmax_topk_mode("auto")
    probs, indices = softmax_topk_cpu_torch(sample_logits, logits_processor=None)
    assert probs.shape == (10,)
    assert indices.shape == (10,)


def test_softmax_topk_respects_custom_max_return_k(sample_logits, restore_softmax_topk_mode):
    """``max_return_k`` controls the size of the returned slice."""
    set_softmax_topk_mode("auto")
    probs, indices = softmax_topk_cpu_torch(sample_logits, max_return_k=5, logits_processor=None)
    assert probs.shape == (5,)
    assert indices.shape == (5,)
