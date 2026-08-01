"""Regression tests: static-vision Qwen3-VL rejects multi-image inputs with a clear error.

Static Qwen3-VL MXQ releases (single-input vision, fixed visual-token count in the
text decoder) bake a single image's 2D RoPE grid into the text decoder. A second
image *for the same prompt* would need its own independent 2D coordinates, which
the baked rope cannot express, so the decoder silently loses the image-boundary
distinction and the language model emits grammatically-plausible but semantically
wrong output. These tests pin down the processor-level guard (which also skips
the image processor's patch extraction) and verify that both the single-image
path and the batched-single-image path are unaffected. The guard is per-prompt,
not total-count: a batch of N single-image prompts must pass through.
"""

from __future__ import annotations

import pytest
from PIL import Image

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


def _make_image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(128, 128, 128))


def test_processor_rejects_multi_image_when_static_vision() -> None:
    """Static-mode processor must raise ``NotImplementedError`` on multi-image input."""
    proc = _make_processor(dynamic_vision=False)
    with pytest.raises(NotImplementedError, match="dynamic-vision Qwen3-VL release"):
        proc(images=[_make_image(), _make_image()], text="describe")


def test_processor_multi_image_guard_fires_before_super_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hard fail must trip before the image processor's patch extraction runs.

    Silent-fail regression was that the processor happily ran ``_resize_images``
    plus the upstream patch extraction, so we guarantee the raise happens before
    ``super().__call__`` is entered — an exploding stub for the upstream call
    verifies we never reach it in the static multi-image path.
    """
    called = {"super": False}

    def _boom_super_call(self, *args, **kwargs):
        called["super"] = True
        raise AssertionError(
            "super().__call__ must not be reached in the static multi-image path"
        )

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _boom_super_call)

    proc = _make_processor(dynamic_vision=False)
    with pytest.raises(NotImplementedError):
        proc(images=[_make_image(), _make_image()], text="describe")
    assert called["super"] is False


def test_processor_lets_multi_image_through_when_dynamic_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic-mode processor keeps forwarding multi-image to the upstream ``__call__``."""
    captured: dict[str, object] = {}

    def _capture_super_call(self, images, text, videos, **kwargs):
        captured["images"] = images
        captured["text"] = text
        captured["videos"] = videos
        return "sentinel-batch-feature"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super_call)

    # Stub `_clamp_dynamic_image_size` since we skipped the heavy __init__ and
    # `image_processor` isn't set — the dynamic path calls it before dispatch.
    def _noop_clamp(self):
        return None

    monkeypatch.setattr(
        MobilintQwen3VLProcessor, "_clamp_dynamic_image_size", _noop_clamp
    )

    proc = _make_processor(dynamic_vision=True)
    imgs = [_make_image(), _make_image()]
    result = proc(images=imgs, text="compare")

    assert result == "sentinel-batch-feature"
    assert captured["images"] is imgs


def test_processor_accepts_single_image_when_static_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: static-mode processor still forwards a single image."""
    captured: dict[str, object] = {}

    def _capture_super_call(self, images, text, videos, **kwargs):
        captured["images"] = images
        captured["text"] = text
        captured["videos"] = videos
        return "sentinel-batch-feature"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super_call)

    # `_resize_images` doesn't need `image_processor`; a raw PIL image is fine.
    proc = _make_processor(dynamic_vision=False)
    single = _make_image()
    result = proc(images=[single], text="describe")

    assert result == "sentinel-batch-feature"
    # `_resize_images` returns a list of resized copies — only the count matters here.
    assert isinstance(captured["images"], list)
    assert len(captured["images"]) == 1


def test_processor_accepts_batched_single_image_when_static_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch of N single-image prompts (``[[img], [img], ...]``) must pass on static.

    The chat-template pipeline nests images as ``[[imgs_prompt_1], ...]``, so a
    batch of N single-image samples arrives with total image count == N but
    per-prompt count == 1. Static-vision MXQ handles this via the batched
    multi-core stack path; only per-prompt multi-image is the constraint.
    """
    captured: dict[str, object] = {}

    def _capture_super_call(self, images, text, videos, **kwargs):
        captured["images"] = images
        captured["text"] = text
        captured["videos"] = videos
        return "sentinel-batch-feature"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super_call)

    proc = _make_processor(dynamic_vision=False)
    batched = [[_make_image()], [_make_image()], [_make_image()]]
    result = proc(images=batched, text=["a", "b", "c"])

    assert result == "sentinel-batch-feature"
    # Each inner list keeps a single (resized) image after `_resize_images`.
    assert isinstance(captured["images"], list)
    assert len(captured["images"]) == 3
    assert all(len(inner) == 1 for inner in captured["images"])


def test_processor_rejects_per_prompt_multi_image_in_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One prompt in a batch with >1 image still trips the static-vision guard."""
    monkeypatch.setattr(
        Qwen3VLProcessor,
        "__call__",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("super().__call__ must not run for per-prompt multi-image")
        ),
    )
    proc = _make_processor(dynamic_vision=False)
    batched = [[_make_image()], [_make_image(), _make_image()]]
    with pytest.raises(NotImplementedError, match="dynamic-vision Qwen3-VL release"):
        proc(images=batched, text=["a", "b"])
