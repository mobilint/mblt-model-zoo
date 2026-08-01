"""Regression tests: ``_resize_one`` handles a 4-D NumPy batch in either layout.

``ImageInput`` allows a 4-D NumPy batch in both ``(N, H, W, C)`` (NHWC) and
``(N, C, H, W)`` (NCHW) layouts. The override's ndarray branch splits along
the batch axis and hands each 3-D frame to ``cv2.resize``, which only
handles HWC frames — so a channels-first frame must be transposed to HWC
before the resize and restored after. These tests pin down that both
layouts survive the resize with correct shape *and* correct channel content.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # noqa: E402

from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLProcessor,
    MobilintQwen3VLVideoProcessor,
)


def _make_processor(dynamic_vision: bool) -> MobilintQwen3VLProcessor:
    """Build a bare processor without going through the heavy ``__init__``."""
    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    proc.dynamic_vision = dynamic_vision
    return proc


def test_resize_one_handles_4d_ndarray_batch() -> None:
    batch = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    resized = MobilintQwen3VLProcessor._resize_one(batch, size=(128, 128))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (3, 128, 128, 3)


def test_resize_one_handles_4d_ndarray_batch_nhwc_preserves_channels() -> None:
    """NHWC regression: a distinguishable per-channel pattern must survive.

    A constant per-channel value survives bicubic resize exactly, so we can
    read the output channels back and verify the layout wasn't scrambled.
    """
    batch = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    batch[..., 0] = 10
    batch[..., 1] = 20
    batch[..., 2] = 30

    resized = MobilintQwen3VLProcessor._resize_one(batch, size=(16, 16))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (2, 16, 16, 3)
    assert int(resized[0, ..., 0].mean()) == 10
    assert int(resized[0, ..., 1].mean()) == 20
    assert int(resized[0, ..., 2].mean()) == 30


def test_resize_one_handles_4d_ndarray_batch_nchw() -> None:
    """NCHW ``(N, C, H, W)`` batch must resize to ``(N, C, target_h, target_w)``.

    Fills each channel with a distinct constant so we can verify the channel
    axis was preserved (not folded into the spatial dim by cv2 misreading a
    ``(C, H, W)`` frame as HWC).
    """
    batch = np.zeros((2, 3, 32, 40), dtype=np.uint8)
    batch[:, 0] = 10
    batch[:, 1] = 20
    batch[:, 2] = 30

    resized = MobilintQwen3VLProcessor._resize_one(batch, size=(16, 24))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (2, 3, 16, 24)
    assert int(resized[0, 0].mean()) == 10
    assert int(resized[0, 1].mean()) == 20
    assert int(resized[0, 2].mean()) == 30
    assert int(resized[1, 0].mean()) == 10
    assert int(resized[1, 1].mean()) == 20
    assert int(resized[1, 2].mean()) == 30


def test_resize_images_handles_4d_ndarray_batch_nchw() -> None:
    batch = np.zeros((2, 3, 100, 100), dtype=np.uint8)
    resized = MobilintQwen3VLProcessor._resize_images(batch)
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (2, 3, 224, 224)


def test_resize_one_preserves_3d_ndarray_behavior() -> None:
    """Guard the existing single-image path used by all other tests."""
    single = np.zeros((256, 256, 3), dtype=np.uint8)
    resized = MobilintQwen3VLProcessor._resize_one(single, size=(224, 224))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (224, 224, 3)


def test_resize_one_handles_3d_ndarray_hwc_preserves_channels() -> None:
    """3-D ``(H, W, C)`` ndarray must resize with channel content intact."""
    img = np.zeros((32, 40, 3), dtype=np.uint8)
    img[..., 0] = 10
    img[..., 1] = 20
    img[..., 2] = 30

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (16, 24, 3)
    assert int(resized[..., 0].mean()) == 10
    assert int(resized[..., 1].mean()) == 20
    assert int(resized[..., 2].mean()) == 30


def test_resize_one_handles_3d_ndarray_chw_preserves_channels() -> None:
    """3-D ``(C, H, W)`` ndarray must resize with channels-first layout intact.

    Previously fed straight to ``cv2.resize`` (which reads axis -1 as the
    channel dim), which corrupts a channels-first frame — a ``(3, 32, 40)``
    array was interpreted as HWC with 40 channels and the spatial dims were
    silently mangled. Pin down that the CHW layout survives the resize.
    """
    img = np.zeros((3, 32, 40), dtype=np.uint8)
    img[0] = 10
    img[1] = 20
    img[2] = 30

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (3, 16, 24)
    assert int(resized[0].mean()) == 10
    assert int(resized[1].mean()) == 20
    assert int(resized[2].mean()) == 30


def test_resize_one_handles_3d_ndarray_hwc_grayscale() -> None:
    """3-D ``(H, W, 1)`` grayscale ndarray must retain the singleton channel.

    ``cv2.resize`` collapses the trailing axis on a single-channel HWC frame
    and returns rank-2 output. Without a rank restore step, ``np.stack`` +
    ``restore`` would either raise during the layout transpose or silently
    return the wrong shape. Pin down that the round-trip layout is
    ``(H', W', 1)``.
    """
    img = np.zeros((32, 40, 1), dtype=np.uint8)
    img[..., 0] = 42

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (16, 24, 1)
    assert int(resized[..., 0].mean()) == 42


def test_resize_one_handles_3d_ndarray_chw_grayscale() -> None:
    """3-D ``(1, H, W)`` grayscale ndarray must retain the leading singleton.

    A CHW single-channel frame is transposed to ``(H, W, 1)`` before cv2,
    which then collapses the trailing axis; the restore step must
    re-materialize both the channel axis and the CHW orientation.
    """
    img = np.zeros((1, 32, 40), dtype=np.uint8)
    img[0] = 42

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(16, 24))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (1, 16, 24)
    assert int(resized[0].mean()) == 42


def test_resize_one_handles_4d_ndarray_bhwc_grayscale() -> None:
    """4-D ``(N, H, W, 1)`` grayscale batch must preserve the singleton channel."""
    batch = np.zeros((3, 32, 40, 1), dtype=np.uint8)
    batch[0, ..., 0] = 10
    batch[1, ..., 0] = 20
    batch[2, ..., 0] = 30

    resized = MobilintQwen3VLProcessor._resize_one(batch, size=(16, 24))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (3, 16, 24, 1)
    assert int(resized[0, ..., 0].mean()) == 10
    assert int(resized[1, ..., 0].mean()) == 20
    assert int(resized[2, ..., 0].mean()) == 30


def test_resize_one_handles_4d_ndarray_bchw_grayscale() -> None:
    """4-D ``(N, 1, H, W)`` grayscale batch must preserve the leading singleton."""
    batch = np.zeros((3, 1, 32, 40), dtype=np.uint8)
    batch[0, 0] = 10
    batch[1, 0] = 20
    batch[2, 0] = 30

    resized = MobilintQwen3VLProcessor._resize_one(batch, size=(16, 24))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (3, 1, 16, 24)
    assert int(resized[0, 0].mean()) == 10
    assert int(resized[1, 0].mean()) == 20
    assert int(resized[2, 0].mean()) == 30


def test_resize_one_ambiguous_3d_ndarray_ties_to_hwc() -> None:
    """Ambiguous ``(3, 3, 3)`` ndarray: tie-break to HWC.

    Both axis 0 and axis -1 have size 3, so both look like plausible channel
    counts. The tie-break policy matches the torch branch and
    ``_count_images``: treat as HWC, keeping the channel axis at -1.
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[..., 0] = 1
    img[..., 1] = 2
    img[..., 2] = 3

    resized = MobilintQwen3VLProcessor._resize_one(img, size=(6, 6))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (6, 6, 3)
    assert int(resized[..., 0].mean()) == 1
    assert int(resized[..., 1].mean()) == 2
    assert int(resized[..., 2].mean()) == 3


def test_resize_images_handles_4d_ndarray_batch() -> None:
    batch = np.zeros((2, 100, 100, 3), dtype=np.uint8)
    resized = MobilintQwen3VLProcessor._resize_images(batch)
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (2, 224, 224, 3)


def test_processor_forwards_4d_ndarray_batch_when_dynamic_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full ``__call__`` accepts a 4-D NumPy batch in dynamic-vision mode.

    Static mode would trip the per-prompt multi-image guard because a 4-D
    batch counts as ``N`` images per prompt, so exercise the dynamic path
    that actually flows through ``_resize_images``.
    """
    captured: dict[str, object] = {}

    def _capture_super_call(self, images, text, videos, **kwargs):
        captured["images"] = images
        return "sentinel-batch-feature"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super_call)
    # Stub both dynamic-vision image clamps: the dynamic path calls
    # ``_clamp_dynamic_image_size`` on stored defaults and
    # ``_clamp_dynamic_image_call_kwargs`` on caller overrides, and neither
    # can inspect ``image_processor`` here because we skipped the heavy
    # ``__init__``.
    monkeypatch.setattr(MobilintQwen3VLProcessor, "_clamp_dynamic_image_size", lambda self: None)
    monkeypatch.setattr(
        MobilintQwen3VLProcessor,
        "_clamp_dynamic_image_call_kwargs",
        lambda self, kwargs: None,
    )

    proc = _make_processor(dynamic_vision=True)
    batch = np.zeros((3, 320, 320, 3), dtype=np.uint8)
    result = proc(images=batch, text="describe")

    assert result == "sentinel-batch-feature"
    # Dynamic-vision skips the forced resize, so the batch flows through
    # untouched — the important property is that it did not crash on the
    # ndarray branch of ``_resize_one`` before dispatch.
    assert captured["images"] is batch
