"""Regression tests: caller cannot override the vision MXQ's structural knobs.

``patch_size`` / ``temporal_patch_size`` / ``merge_size`` are baked into the
vision MXQ at compile time — the folded feature width the language model
expects at the vision-language boundary is ``patch_size * merge_size`` and the
temporal stride is ``temporal_patch_size``. ``Qwen3VLImagesKwargs`` and
``Qwen3VLVideosKwargs`` still surface them as call-time overrides, and the
token-budget clamp derives its ceiling from the *stored* ``patch_size``, so a
caller-supplied smaller value both breaks the boundary shape and silently
bypasses the NPU vision-token guard.

These tests exercise ``MobilintQwen3VLProcessor.__call__`` for both modalities
and both scopes (top-level and nested ``images_kwargs`` / ``videos_kwargs``).
"""

from __future__ import annotations

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


class _ImageProcessorStub:
    """Minimal image processor exposing the structural attributes the guard reads."""

    def __init__(
        self,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
    ) -> None:
        self.size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size


def _make_processor(dynamic_vision: bool = True) -> MobilintQwen3VLProcessor:
    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.image_processor = _ImageProcessorStub()
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    proc.dynamic_vision = dynamic_vision
    return proc


# ---------------------------------------------------------------------------
# Image path — direct method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_image_structural_override_top_level_raises(field: str) -> None:
    """A top-level structural override that differs from the baseline raises."""
    proc = _make_processor()
    baseline = getattr(proc.image_processor, field)
    kwargs: dict = {field: baseline + 1}

    with pytest.raises(ValueError, match=field):
        proc._reject_structural_vision_overrides(kwargs, "images_kwargs", "image_processor", "image")


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_image_structural_override_nested_raises(field: str) -> None:
    """The nested ``images_kwargs`` route is caught the same way."""
    proc = _make_processor()
    baseline = getattr(proc.image_processor, field)
    kwargs: dict = {"images_kwargs": {field: baseline + 1}}

    with pytest.raises(ValueError, match=field):
        proc._reject_structural_vision_overrides(kwargs, "images_kwargs", "image_processor", "image")


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_image_structural_override_equal_to_default_is_noop(field: str) -> None:
    """A caller-supplied value that matches the baseline is silently popped."""
    proc = _make_processor()
    baseline = getattr(proc.image_processor, field)
    kwargs: dict = {
        field: baseline,
        "images_kwargs": {field: baseline},
    }

    proc._reject_structural_vision_overrides(kwargs, "images_kwargs", "image_processor", "image")

    assert field not in kwargs
    assert field not in kwargs["images_kwargs"]


def test_image_structural_none_override_is_noop() -> None:
    """A caller passing ``None`` is treated as "no override" and popped without raising."""
    proc = _make_processor()
    kwargs: dict = {
        "patch_size": None,
        "images_kwargs": {"merge_size": None},
    }

    proc._reject_structural_vision_overrides(kwargs, "images_kwargs", "image_processor", "image")

    assert "patch_size" not in kwargs
    assert "merge_size" not in kwargs["images_kwargs"]


# ---------------------------------------------------------------------------
# Image path — end-to-end via ``__call__``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_call_image_structural_override_hard_fails_before_super_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """A structural override on ``__call__`` raises before ``super().__call__`` runs."""
    reached = {"super": False}

    def _boom_super(self, *args, **kwargs):
        reached["super"] = True
        raise AssertionError("super().__call__ must not run when a structural override is set")

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _boom_super)

    proc = _make_processor()
    baseline = getattr(proc.image_processor, field)
    with pytest.raises(ValueError, match=field):
        proc(
            images=[object()],
            text="describe <|image_pad|>",
            images_kwargs={field: baseline + 1},
        )
    assert reached["super"] is False


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_call_image_structural_override_equal_default_passes(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """A caller-supplied value equal to the shipped default forwards cleanly."""
    proc = _make_processor()
    baseline = getattr(proc.image_processor, field)
    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    result = proc(
        images=[object()],
        text="describe <|image_pad|>",
        images_kwargs={field: baseline},
    )

    assert result == "sentinel"
    forwarded = captured["kwargs"]
    # The value matched the baseline and was popped before dispatch, so the
    # upstream image path uses the shipped default.
    assert field not in forwarded.get("images_kwargs", {})


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_call_image_structural_override_static_mode_also_hard_fails(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """The guard also runs on the static-vision image path.

    Static releases compile the vision MXQ with the same structural knobs, so
    the reject must not be gated on ``dynamic_vision``.
    """
    reached = {"super": False}

    def _boom_super(self, *args, **kwargs):
        reached["super"] = True
        raise AssertionError("super().__call__ must not run when a structural override is set")

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _boom_super)

    proc = _make_processor(dynamic_vision=False)
    baseline = getattr(proc.image_processor, field)

    from PIL import Image

    image = Image.new("RGB", (224, 224))
    with pytest.raises(ValueError, match=field):
        proc(
            images=[image],
            text="describe <|image_pad|>",
            images_kwargs={field: baseline + 1},
        )
    assert reached["super"] is False


# ---------------------------------------------------------------------------
# Video path — direct method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_video_structural_override_top_level_raises(field: str) -> None:
    """A top-level video structural override that differs from the baseline raises."""
    proc = _make_processor()
    baseline = getattr(proc.video_processor, field)
    kwargs: dict = {field: baseline + 1}

    with pytest.raises(ValueError, match=field):
        proc._reject_structural_vision_overrides(kwargs, "videos_kwargs", "video_processor", "video")


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_video_structural_override_nested_raises(field: str) -> None:
    """``videos_kwargs={<field>: ...}`` is caught in the nested dict."""
    proc = _make_processor()
    baseline = getattr(proc.video_processor, field)
    kwargs: dict = {"videos_kwargs": {field: baseline + 1}}

    with pytest.raises(ValueError, match=field):
        proc._reject_structural_vision_overrides(kwargs, "videos_kwargs", "video_processor", "video")


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_video_structural_override_equal_to_default_is_noop(field: str) -> None:
    """A caller-supplied video value that matches the baseline is silently popped."""
    proc = _make_processor()
    baseline = getattr(proc.video_processor, field)
    kwargs: dict = {
        field: baseline,
        "videos_kwargs": {field: baseline},
    }

    proc._reject_structural_vision_overrides(kwargs, "videos_kwargs", "video_processor", "video")

    assert field not in kwargs
    assert field not in kwargs["videos_kwargs"]


# ---------------------------------------------------------------------------
# Video path — end-to-end via ``__call__``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_call_video_structural_override_hard_fails_before_super_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """A structural override on the video path raises before ``super().__call__`` runs."""
    reached = {"super": False}

    def _boom_super(self, *args, **kwargs):
        reached["super"] = True
        raise AssertionError("super().__call__ must not run when a structural override is set")

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _boom_super)

    proc = _make_processor()
    baseline = getattr(proc.video_processor, field)
    with pytest.raises(ValueError, match=field):
        proc(
            images=None,
            text="describe <|video_pad|>",
            videos=[object()],
            videos_kwargs={field: baseline + 1},
        )
    assert reached["super"] is False


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_call_video_structural_override_equal_default_passes(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """A caller-supplied video value equal to the shipped default forwards cleanly."""
    proc = _make_processor()
    baseline = getattr(proc.video_processor, field)
    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    result = proc(
        images=None,
        text="describe <|video_pad|>",
        videos=[object()],
        videos_kwargs={field: baseline},
    )

    assert result == "sentinel"
    forwarded = captured["kwargs"]
    assert field not in forwarded.get("videos_kwargs", {})
