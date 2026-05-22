"""Unit tests for Mobilint cache snapshot helpers."""

from __future__ import annotations

import pytest
import torch

from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache, MobilintDeepStackCache


class _FakeMxqModel:
    """Minimal MXQ stub for cache snapshot tests."""

    def __init__(self) -> None:
        self.loaded: list[tuple[int, list[bytes]]] = []

    def dump_cache_memory(self, cache_id: int) -> list[bytes]:
        """Return a stable in-memory cache payload for the requested cache id."""
        return [f"cache-{cache_id}".encode("utf-8")]

    def load_cache_memory(self, buffer: list[bytes], cache_id: int) -> None:
        """Record the cache payload that was restored into the fake backend."""
        self.loaded.append((cache_id, list(buffer)))


def test_dump_cache_memory_roundtrip_restores_seq_length() -> None:
    """Restore in-memory cache snapshots with their original sequence length."""
    mxq_model = _FakeMxqModel()
    cache = MobilintCache(mxq_model)

    cache.set_seq_length(7)
    cache.dump_cache_memory()
    cache.set_seq_length(0)

    cache.load_cache_memory()

    assert cache.get_seq_length() == 7
    assert mxq_model.loaded == [(0, [b"cache-0"])]


def test_fake_prefill_sets_seq_length_without_cache_buffer() -> None:
    """Fake prefill should only expose sequence length and clear cache payloads."""
    mxq_model = _FakeMxqModel()
    cache = MobilintCache(mxq_model)

    cache.set_seq_length(7)
    cache.dump_cache_memory()

    cache.fake_prefill(128)

    assert cache.get_seq_length() == 128
    assert cache.layers[0].buffer == []
    assert cache.layers[0].buffer_seq_length is None
    assert mxq_model.loaded == []


def test_fake_prefill_sets_batched_seq_lengths() -> None:
    """Fake prefill should support per-cache-id lengths for batched decode."""
    cache = MobilintCache(_FakeMxqModel(), batch_size=2)

    cache.fake_prefill({0: 32, 1: 64})

    assert cache.get_seq_length(index=0) == 32
    assert cache.get_seq_length(index=1) == 64


def test_fake_prefill_scalar_sets_all_batched_seq_lengths() -> None:
    """Scalar fake prefill should prepare every cache entry for batched decode."""
    cache = MobilintCache(_FakeMxqModel(), batch_size=3)

    cache.fake_prefill(128)

    assert [cache.get_seq_length(index=i) for i in range(3)] == [128, 128, 128]


def test_fake_prefill_rejects_negative_length() -> None:
    """Fake prefill should reuse sequence length validation."""
    cache = MobilintCache(_FakeMxqModel())

    with pytest.raises(ValueError, match="non-negative"):
        cache.fake_prefill(-1)


def test_deepstack_cache_returns_real_chunk() -> None:
    """Deepstack cache should slice the current forward-call tensor."""
    cache = MobilintDeepStackCache(_FakeMxqModel(), num_deepstack_layers=2, hidden_size=3)
    deepstack = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)

    cache.set_deepstack_tensor(deepstack)

    chunk = cache.get_deepstack_chunk(1, 3, device=torch.device("cpu"), dtype=torch.float32)

    assert torch.equal(chunk, deepstack[:, 1:3, :])


def test_deepstack_cache_fake_prefill_returns_zero_chunk() -> None:
    """Fake-prefilled deepstack cache should lazily provide zero decode chunks."""
    cache = MobilintDeepStackCache(_FakeMxqModel(), num_deepstack_layers=2, hidden_size=3)

    cache.fake_prefill(128)
    chunk = cache.get_deepstack_chunk(0, 1, device=torch.device("cpu"), dtype=torch.float32)

    assert cache.get_seq_length() == 128
    assert chunk.shape == (2, 1, 3)
    assert torch.count_nonzero(chunk).item() == 0


def test_deepstack_cache_reset_clears_deepstack_tensor() -> None:
    """Reset should clear per-call deepstack payloads and sequence length."""
    cache = MobilintDeepStackCache(_FakeMxqModel(), num_deepstack_layers=1, hidden_size=2)
    cache.set_deepstack_tensor(torch.ones(1, 2, 2))
    cache.set_seq_length(4)

    cache.reset()
    chunk = cache.get_deepstack_chunk(0, 2, device=torch.device("cpu"), dtype=torch.float32)

    assert cache.get_seq_length() == 0
    assert torch.count_nonzero(chunk).item() == 0
