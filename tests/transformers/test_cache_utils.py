"""Unit tests for Mobilint cache snapshot helpers."""

from __future__ import annotations

from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache


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
