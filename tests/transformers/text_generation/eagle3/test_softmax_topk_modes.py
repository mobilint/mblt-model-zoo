"""Unit tests for the ``softmax_topk_cpu_torch`` A/B mode switch."""

from __future__ import annotations

import importlib

import pytest
import torch

from mblt_model_zoo.hf_transformers.utils.eagle3 import tree_decoding as tree_decoding_module
from mblt_model_zoo.hf_transformers.utils.eagle3.tree_decoding import (
    _VALID_SOFTMAX_TOPK_MODES,
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


def test_default_mode_is_sliced():
    """The default mode is the sliced renormalized softmax path."""
    reloaded = importlib.reload(tree_decoding_module)
    try:
        assert reloaded.SOFTMAX_TOPK_MODE == "sliced"
    finally:
        importlib.reload(tree_decoding_module)


def test_env_var_selects_full_mode(monkeypatch):
    """A ``full`` env var value opts back into the legacy full-vocab softmax at import time."""
    monkeypatch.setenv("MBLT_EAGLE3_SOFTMAX_TOPK_MODE", "full")
    reloaded = importlib.reload(tree_decoding_module)
    try:
        assert reloaded.SOFTMAX_TOPK_MODE == "full"
    finally:
        monkeypatch.delenv("MBLT_EAGLE3_SOFTMAX_TOPK_MODE", raising=False)
        importlib.reload(tree_decoding_module)


def test_env_var_invalid_falls_back_to_sliced(monkeypatch):
    """An unrecognized env var value falls back to the ``sliced`` default."""
    monkeypatch.setenv("MBLT_EAGLE3_SOFTMAX_TOPK_MODE", "not-a-mode")
    reloaded = importlib.reload(tree_decoding_module)
    try:
        assert reloaded.SOFTMAX_TOPK_MODE == "sliced"
    finally:
        monkeypatch.delenv("MBLT_EAGLE3_SOFTMAX_TOPK_MODE", raising=False)
        importlib.reload(tree_decoding_module)


def test_set_softmax_topk_mode_rejects_unknown(restore_softmax_topk_mode):
    """The programmatic setter rejects unsupported modes."""
    with pytest.raises(ValueError):
        set_softmax_topk_mode("bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", _VALID_SOFTMAX_TOPK_MODES)
def test_both_modes_return_same_topk_indices_no_processor(mode, sample_logits, restore_softmax_topk_mode):
    """Both modes must select the same top-k indices when no processor is provided."""
    set_softmax_topk_mode("full")
    _, full_idx = softmax_topk_cpu_torch(sample_logits, 10, logits_processor=None)
    set_softmax_topk_mode(mode)
    _, mode_idx = softmax_topk_cpu_torch(sample_logits, 10, logits_processor=None)
    assert torch.equal(full_idx, mode_idx)


def test_sliced_probs_sum_to_one(sample_logits, restore_softmax_topk_mode):
    """The ``sliced`` mode returns probabilities that sum to 1.0 across the slice."""
    set_softmax_topk_mode("sliced")
    probs, _ = softmax_topk_cpu_torch(sample_logits, 10, logits_processor=None)
    assert probs.shape == (10,)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_full_probs_sum_below_one(sample_logits, restore_softmax_topk_mode):
    """The ``full`` mode returns partial softmax mass (sum strictly below 1.0)."""
    set_softmax_topk_mode("full")
    probs, _ = softmax_topk_cpu_torch(sample_logits, 10, logits_processor=None)
    total = probs.sum().item()
    assert total <= 1.0
    assert total < 1.0 - 1e-6  # a diffuse random distribution leaves mass outside the top-10


def test_both_modes_agree_on_topk_indices_with_temperature(sample_logits, restore_softmax_topk_mode):
    """Applying Temperature keeps the top-k selection identical across modes."""
    processor = prepare_logits_processor(temperature=0.7)
    assert processor is not None
    set_softmax_topk_mode("full")
    _, full_idx = softmax_topk_cpu_torch(sample_logits, 10, logits_processor=processor)
    set_softmax_topk_mode("sliced")
    _, sliced_idx = softmax_topk_cpu_torch(sample_logits, 10, logits_processor=processor)
    assert torch.equal(full_idx, sliced_idx)


def test_sliced_matches_full_softmax_on_slice_after_temperature(sample_logits, restore_softmax_topk_mode):
    """Sliced probabilities equal full-vocab softmax over the same slice, renormalized."""
    processor = prepare_logits_processor(temperature=0.7)
    assert processor is not None
    set_softmax_topk_mode("full")
    full_probs, full_idx = softmax_topk_cpu_torch(sample_logits, 10, logits_processor=processor)
    set_softmax_topk_mode("sliced")
    sliced_probs, sliced_idx = softmax_topk_cpu_torch(sample_logits, 10, logits_processor=processor)
    assert torch.equal(full_idx, sliced_idx)
    expected = full_probs / full_probs.sum()
    assert torch.allclose(sliced_probs, expected, atol=1e-6)


def test_sliced_mode_supports_2d_logits(restore_softmax_topk_mode):
    """The 2D logits path returns per-row probabilities that sum to 1.0."""
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 128, generator=generator, dtype=torch.float32)
    set_softmax_topk_mode("sliced")
    probs, indices = softmax_topk_cpu_torch(logits, 10, logits_processor=None)
    assert probs.shape == (3, 10)
    assert indices.shape == (3, 10)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
