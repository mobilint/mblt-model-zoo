"""Regression tests: ``_resize_one`` handles torch tensors in either channel layout.

``ImageInput`` allows a 3-D torch tensor in either ``(H, W, C)`` or
``(C, H, W)`` layout and a 4-D tensor in either ``(N, H, W, C)`` or
``(N, C, H, W)`` layout. ``F.interpolate`` treats its input as NCHW
unconditionally, so an HWC / BHWC tensor previously had its channel axis
silently bicubic-resized as if it were height. Companion to
``test_qwen3_vl_resize_ndarray_batch.py`` — pin down that both layouts
survive the resize with correct shape *and* correct channel content, and
that the channels-first regressions still work.
"""

from __future__ import annotations

import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLProcessor,
)


def test_resize_one_handles_3d_torch_tensor_hwc_preserves_channels() -> None:
    """3-D ``(H, W, C)`` tensor must resize to ``(target_h, target_w, C)``.

    Fill each channel with a distinct constant so we can verify the
    channel axis was preserved (not folded into the spatial dim by
    ``F.interpolate`` misreading a ``(H, W, C)`` tensor as CHW). A
    constant per-channel value survives bicubic resize exactly.
    """
    img = torch.zeros((32, 40, 3), dtype=torch.float32)
    img[..., 0] = 10.0
    img[..., 1] = 20.0
    img[..., 2] = 30.0

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, torch.Tensor)
    assert tuple(resized.shape) == (16, 24, 3)
    assert torch.allclose(resized[..., 0], torch.full((16, 24), 10.0))
    assert torch.allclose(resized[..., 1], torch.full((16, 24), 20.0))
    assert torch.allclose(resized[..., 2], torch.full((16, 24), 30.0))


def test_resize_one_handles_4d_torch_tensor_bhwc_preserves_channels() -> None:
    """4-D ``(N, H, W, C)`` batch must resize to ``(N, target_h, target_w, C)``.

    Verifies per-frame that the channel axis survived the resize + permute
    round-trip; the previous code fed BHWC straight to ``F.interpolate``
    (which reads axis 1 as channels) and silently produced garbage.
    """
    batch = torch.zeros((2, 32, 40, 3), dtype=torch.float32)
    batch[..., 0] = 10.0
    batch[..., 1] = 20.0
    batch[..., 2] = 30.0

    resized = MobilintQwen3VLProcessor._resize_one(batch, size=(16, 24))
    assert isinstance(resized, torch.Tensor)
    assert tuple(resized.shape) == (2, 16, 24, 3)
    for i in range(2):
        assert torch.allclose(resized[i, ..., 0], torch.full((16, 24), 10.0))
        assert torch.allclose(resized[i, ..., 1], torch.full((16, 24), 20.0))
        assert torch.allclose(resized[i, ..., 2], torch.full((16, 24), 30.0))


def test_resize_one_handles_3d_torch_tensor_chw_regression() -> None:
    """CHW regression: ``(C, H, W)`` still resizes correctly (no layout change)."""
    img = torch.zeros((3, 32, 40), dtype=torch.float32)
    img[0] = 10.0
    img[1] = 20.0
    img[2] = 30.0

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, torch.Tensor)
    assert tuple(resized.shape) == (3, 16, 24)
    assert torch.allclose(resized[0], torch.full((16, 24), 10.0))
    assert torch.allclose(resized[1], torch.full((16, 24), 20.0))
    assert torch.allclose(resized[2], torch.full((16, 24), 30.0))


def test_resize_one_handles_4d_torch_tensor_bchw_regression() -> None:
    """BCHW regression: ``(N, C, H, W)`` still resizes correctly."""
    batch = torch.zeros((2, 3, 32, 40), dtype=torch.float32)
    batch[:, 0] = 10.0
    batch[:, 1] = 20.0
    batch[:, 2] = 30.0

    resized = MobilintQwen3VLProcessor._resize_one(batch, size=(16, 24))
    assert isinstance(resized, torch.Tensor)
    assert tuple(resized.shape) == (2, 3, 16, 24)
    for i in range(2):
        assert torch.allclose(resized[i, 0], torch.full((16, 24), 10.0))
        assert torch.allclose(resized[i, 1], torch.full((16, 24), 20.0))
        assert torch.allclose(resized[i, 2], torch.full((16, 24), 30.0))


def test_resize_one_ambiguous_3d_torch_tensor_ties_to_hwc() -> None:
    """Ambiguous ``(3, H, W)`` where H and W also look like channels: tie-break to HWC.

    A ``(3, 3, 4)`` tensor has both axis 0 and axis -1 in ``{1, 3, 4}``. The
    tie-break policy matches the ndarray branch and the majority upstream
    convention: treat as HWC. That means the channel axis is axis -1 (size
    4) and the spatial dims are (3, 3).
    """
    img = torch.zeros((3, 3, 4), dtype=torch.float32)
    img[..., 0] = 1.0
    img[..., 1] = 2.0
    img[..., 2] = 3.0
    img[..., 3] = 4.0

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(6, 6))
    assert isinstance(resized, torch.Tensor)
    # HWC tie-break: channel axis stays at -1, size 4.
    assert tuple(resized.shape) == (6, 6, 4)
    assert torch.allclose(resized[..., 0], torch.full((6, 6), 1.0))
    assert torch.allclose(resized[..., 3], torch.full((6, 6), 4.0))
