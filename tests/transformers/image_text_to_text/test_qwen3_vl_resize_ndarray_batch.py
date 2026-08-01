"""Regression tests: ``_resize_one`` handles a 4-D ``(N, H, W, C)`` NumPy batch.

``ImageInput`` allows a 4-D NumPy batch, and the upstream Qwen3-VL processor
unrolls such an array into per-frame images before its own resize. Our
override's ndarray branch previously called ``cv2.resize`` directly, which
only handles 2-D or 3-D inputs and fails on 4-D batches. This test pins down
the fix: iterate along the batch axis, resize each frame, and re-stack —
mirroring the tensor branch, where ``F.interpolate`` preserves the batch dim
for 4-D input natively.
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


def test_resize_one_preserves_3d_ndarray_behavior() -> None:
    """Guard the existing single-image path used by all other tests."""
    single = np.zeros((256, 256, 3), dtype=np.uint8)
    resized = MobilintQwen3VLProcessor._resize_one(single, size=(224, 224))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (224, 224, 3)


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
    monkeypatch.setattr(
        MobilintQwen3VLProcessor, "_clamp_dynamic_image_size", lambda self: None
    )

    proc = _make_processor(dynamic_vision=True)
    batch = np.zeros((3, 320, 320, 3), dtype=np.uint8)
    result = proc(images=batch, text="describe")

    assert result == "sentinel-batch-feature"
    # Dynamic-vision skips the forced resize, so the batch flows through
    # untouched — the important property is that it did not crash on the
    # ndarray branch of ``_resize_one`` before dispatch.
    assert captured["images"] is batch
