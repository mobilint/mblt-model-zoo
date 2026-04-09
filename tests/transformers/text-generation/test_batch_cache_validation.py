import pytest

from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin


class _FakeCache:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size


def test_validate_batch_cache_accepts_matching_size():
    MobilintModelMixin._validate_batch_cache(_FakeCache(batch_size=4), batch_size=4)


def test_validate_batch_cache_accepts_larger_size():
    MobilintModelMixin._validate_batch_cache(_FakeCache(batch_size=8), batch_size=4)


def test_validate_batch_cache_rejects_smaller_size():
    with pytest.raises(ValueError, match="Batch cache size is too small"):
        MobilintModelMixin._validate_batch_cache(_FakeCache(batch_size=1), batch_size=4)
