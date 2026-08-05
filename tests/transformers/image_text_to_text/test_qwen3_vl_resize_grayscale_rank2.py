"""Regression tests: ``_resize_one`` accepts rank-2 grayscale inputs.

``ImageInput`` allows a rank-2 ``(H, W)`` grayscale frame in both NumPy and
Torch, and callers on the static-vision path routinely pass raw grayscale
that has not been promoted to a singleton channel. ``_to_bhwc`` deliberately
rejects rank-2, so ``_resize_one`` must promote to ``(H, W, 1)`` before
handing off and squeeze back to ``(H', W')`` on the way out. Companion to
``test_qwen3_vl_resize_ndarray_batch.py`` and
``test_qwen3_vl_resize_torch_tensor.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLProcessor,
)


def test_resize_one_handles_2d_ndarray_grayscale() -> None:
    """Rank-2 ``(H, W)`` ndarray must resize to ``(H', W')`` with values intact."""
    img = np.zeros((32, 40), dtype=np.float32)
    img[:] = 42.0

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (16, 24)
    assert float(resized.mean()) == 42.0


def test_resize_one_handles_2d_torch_tensor_grayscale() -> None:
    """Rank-2 ``(H, W)`` torch tensor must resize to ``(H', W')`` with values intact."""
    img = torch.zeros((32, 40), dtype=torch.float32)
    img[:] = 42.0

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, torch.Tensor)
    assert tuple(resized.shape) == (16, 24)
    assert torch.allclose(resized, torch.full((16, 24), 42.0))


def test_resize_one_2d_ndarray_promotion_does_not_shadow_hwc_singleton() -> None:
    """Rank-3 ``(H, W, 1)`` ndarray still round-trips through the existing path."""
    img = np.zeros((32, 40, 1), dtype=np.uint8)
    img[..., 0] = 7

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (16, 24, 1)
    assert int(resized[..., 0].mean()) == 7


def test_resize_one_2d_ndarray_promotion_does_not_shadow_chw_singleton() -> None:
    """Rank-3 ``(1, H, W)`` ndarray still round-trips through the existing path."""
    img = np.zeros((1, 32, 40), dtype=np.uint8)
    img[0] = 7

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (1, 16, 24)
    assert int(resized[0].mean()) == 7


def test_resize_one_2d_torch_promotion_does_not_shadow_hwc_singleton() -> None:
    """Rank-3 ``(H, W, 1)`` torch tensor still round-trips through the existing path."""
    img = torch.zeros((32, 40, 1), dtype=torch.float32)
    img[..., 0] = 7.0

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, torch.Tensor)
    assert tuple(resized.shape) == (16, 24, 1)
    assert torch.allclose(resized[..., 0], torch.full((16, 24), 7.0))


def test_resize_one_2d_torch_promotion_does_not_shadow_chw_singleton() -> None:
    """Rank-3 ``(1, H, W)`` torch tensor still round-trips through the existing path."""
    img = torch.zeros((1, 32, 40), dtype=torch.float32)
    img[0] = 7.0

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, torch.Tensor)
    assert tuple(resized.shape) == (1, 16, 24)
    assert torch.allclose(resized[0], torch.full((16, 24), 7.0))
