"""Regression tests: caller kwargs cannot bypass the NPU vision-token ceiling.

``_clamp_dynamic_image_size`` and ``_clamp_dynamic_video_size`` cap the stored
processor defaults, but the caller can still supply overrides at call time via
top-level kwargs or the nested ``images_kwargs`` / ``videos_kwargs`` slots.
Before this fix, any of the following silently produced a vision-patch grid
above ``max_vision_tokens`` and hung the NPU (watchdog timeout ->
``Model_NotAlive``):

- ``max_pixels=<huge>`` or ``min_pixels=<huge>`` (images path)
- ``size={'longest_edge': <huge>, 'shortest_edge': <huge>}`` on either path
- ``do_resize=False`` (either path — strips the ceiling entirely)

These tests exercise both scopes for both modalities and assert that either
the produced grid stays within the budget or the call raises loudly with a
message that points at the ceiling.
"""

from __future__ import annotations

import pytest
import torch

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
    """Minimal image processor for exercising ``_clamp_dynamic_image_call_kwargs``.

    Provides only the attributes the clamp reads (``size``, ``patch_size``);
    the actual patch extraction is exercised end-to-end via the real upstream
    processor in the token-budget tests below.
    """

    def __init__(self, patch_size: int = 14, merge_size: int = 2) -> None:
        # Same shape as the default Qwen2VL/Qwen3VL image processor default.
        self.size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        self.patch_size = patch_size
        self.merge_size = merge_size


def _make_processor(dynamic_vision: bool = True) -> MobilintQwen3VLProcessor:
    """Build a bare processor without going through the heavy ``__init__``."""
    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.image_processor = _ImageProcessorStub()
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    proc.dynamic_vision = dynamic_vision
    return proc


def _expected_image_limit(proc: MobilintQwen3VLProcessor) -> int:
    ip = proc.image_processor
    return proc.max_vision_tokens * ip.patch_size**2


def _expected_video_limit(proc: MobilintQwen3VLProcessor) -> int:
    vp = proc.video_processor
    return proc.max_vision_tokens * vp.patch_size**2 * vp.temporal_patch_size


# ---------------------------------------------------------------------------
# Image path — direct clamp method
# ---------------------------------------------------------------------------


def test_image_call_kwargs_top_level_max_pixels_capped() -> None:
    """A top-level ``max_pixels`` override above the ceiling is clamped in place."""
    proc = _make_processor()
    limit = _expected_image_limit(proc)
    kwargs: dict = {"max_pixels": limit * 8}

    proc._clamp_dynamic_image_call_kwargs(kwargs)

    assert kwargs["max_pixels"] == limit


def test_image_call_kwargs_top_level_min_pixels_capped() -> None:
    """``min_pixels`` scale-up bypass is also clamped."""
    proc = _make_processor()
    limit = _expected_image_limit(proc)
    kwargs: dict = {"min_pixels": limit * 8}

    proc._clamp_dynamic_image_call_kwargs(kwargs)

    assert kwargs["min_pixels"] == limit


def test_image_call_kwargs_nested_images_kwargs_max_pixels_capped() -> None:
    """The nested ``images_kwargs['max_pixels']`` route is capped in place too."""
    proc = _make_processor()
    limit = _expected_image_limit(proc)
    kwargs: dict = {"images_kwargs": {"max_pixels": limit * 8}}

    proc._clamp_dynamic_image_call_kwargs(kwargs)

    assert kwargs["images_kwargs"]["max_pixels"] == limit


def test_image_call_kwargs_nested_images_kwargs_size_capped() -> None:
    """A caller ``images_kwargs={'size': {...}}`` override is clamped edge-by-edge."""
    proc = _make_processor()
    limit = _expected_image_limit(proc)
    kwargs: dict = {
        "images_kwargs": {
            "size": {"longest_edge": limit * 8, "shortest_edge": limit * 4},
        },
    }

    proc._clamp_dynamic_image_call_kwargs(kwargs)

    clamped = kwargs["images_kwargs"]["size"]
    assert clamped["longest_edge"] == limit
    assert clamped["shortest_edge"] == limit


def test_image_call_kwargs_size_preserves_small_shortest_edge() -> None:
    """A ``shortest_edge`` already inside the budget is not inflated."""
    proc = _make_processor()
    limit = _expected_image_limit(proc)
    kwargs: dict = {
        "images_kwargs": {
            "size": {"longest_edge": limit * 8, "shortest_edge": 3136},
        },
    }

    proc._clamp_dynamic_image_call_kwargs(kwargs)

    clamped = kwargs["images_kwargs"]["size"]
    assert clamped["longest_edge"] == limit
    assert clamped["shortest_edge"] == 3136


def test_image_call_kwargs_top_level_do_resize_false_hard_fails() -> None:
    """``do_resize=False`` strips the ceiling — the guard must raise loudly."""
    proc = _make_processor()
    with pytest.raises(ValueError, match="do_resize=False"):
        proc._clamp_dynamic_image_call_kwargs({"do_resize": False})


def test_image_call_kwargs_nested_do_resize_false_hard_fails() -> None:
    """``images_kwargs={'do_resize': False}`` is caught the same way."""
    proc = _make_processor()
    with pytest.raises(ValueError, match="do_resize=False"):
        proc._clamp_dynamic_image_call_kwargs({"images_kwargs": {"do_resize": False}})


def test_image_call_kwargs_noop_when_within_budget() -> None:
    """Overrides already inside the ceiling are left untouched."""
    proc = _make_processor()
    limit = _expected_image_limit(proc)
    kwargs: dict = {
        "max_pixels": limit // 2,
        "min_pixels": limit // 4,
        "images_kwargs": {
            "size": {"longest_edge": limit // 2, "shortest_edge": limit // 4},
        },
    }
    snapshot = {
        "max_pixels": kwargs["max_pixels"],
        "min_pixels": kwargs["min_pixels"],
        "size": dict(kwargs["images_kwargs"]["size"]),
    }

    proc._clamp_dynamic_image_call_kwargs(kwargs)

    assert kwargs["max_pixels"] == snapshot["max_pixels"]
    assert kwargs["min_pixels"] == snapshot["min_pixels"]
    assert kwargs["images_kwargs"]["size"] == snapshot["size"]


# ---------------------------------------------------------------------------
# Image path — end-to-end via ``__call__``
# ---------------------------------------------------------------------------


def test_call_forwards_clamped_image_kwargs_to_super(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``__call__`` must run the clamp before ``super().__call__`` sees the kwargs."""
    proc = _make_processor()
    limit = _expected_image_limit(proc)
    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    result = proc(
        images=[object()],
        text="describe <|image_pad|>",
        max_pixels=limit * 8,
    )

    assert result == "sentinel"
    forwarded = captured["kwargs"]
    assert forwarded["max_pixels"] == limit


def test_call_do_resize_false_bypass_hard_fails_before_super_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hard fail must trip before ``super().__call__`` is entered.

    Any state the upstream call would touch (loading images, computing the
    tokenized prompt, calling the underlying image processor with the raw
    resolution) must not run once the guard has decided to reject.
    """
    reached = {"super": False}

    def _boom_super(self, *args, **kwargs):
        reached["super"] = True
        raise AssertionError("super().__call__ must not run when do_resize=False")

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _boom_super)

    proc = _make_processor()
    with pytest.raises(ValueError, match="do_resize=False"):
        proc(images=[object()], text="describe <|image_pad|>", do_resize=False)
    assert reached["super"] is False


# ---------------------------------------------------------------------------
# Image path — real image processor produces a grid inside the budget
# ---------------------------------------------------------------------------


def _make_processor_with_real_image_processor(dynamic_vision: bool = True) -> MobilintQwen3VLProcessor:
    """Build a processor whose ``image_processor`` is the real upstream implementation.

    Exercises the actual ``smart_resize`` -> patch-count path so the token
    budget assertion covers the end-to-end effect of the clamp, not just its
    bookkeeping.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.image_processor = Qwen2VLImageProcessor()
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    proc.dynamic_vision = dynamic_vision
    return proc


def test_huge_max_pixels_still_produces_within_budget_image_grid() -> None:
    """A ``max_pixels`` override + 4K synthetic image yields a grid inside the ceiling.

    Runs the real image processor after the clamp so an oversized
    ``max_pixels`` cannot expand ``smart_resize`` past the vision-token budget.
    """
    proc = _make_processor_with_real_image_processor()
    ip = proc.image_processor
    limit = proc.max_vision_tokens * ip.patch_size**2

    kwargs: dict = {"max_pixels": limit * 8}
    proc._clamp_dynamic_image_call_kwargs(kwargs)

    image = torch.zeros((3, 2160, 3840), dtype=torch.uint8).numpy()
    result = ip(images=[image], **kwargs)
    grid_thw = result["image_grid_thw"]
    per_image_tokens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]

    assert bool((per_image_tokens > 0).all())
    assert bool((per_image_tokens <= proc.max_vision_tokens).all()), (
        f"per-image grid exceeded budget: {per_image_tokens.tolist()} > {proc.max_vision_tokens}"
    )


# ---------------------------------------------------------------------------
# Video path — direct clamp method
# ---------------------------------------------------------------------------


def test_video_call_kwargs_top_level_size_capped() -> None:
    """A top-level ``size={...}`` override above the video ceiling is clamped."""
    proc = _make_processor()
    limit = _expected_video_limit(proc)
    kwargs: dict = {"size": {"longest_edge": limit * 8, "shortest_edge": limit * 4}}

    proc._clamp_dynamic_video_call_kwargs(kwargs)

    clamped = kwargs["size"]
    assert clamped["longest_edge"] == limit
    assert clamped["shortest_edge"] == limit


def test_video_call_kwargs_nested_videos_kwargs_size_capped() -> None:
    """``videos_kwargs={'size': {...}}`` is capped in the nested dict."""
    proc = _make_processor()
    limit = _expected_video_limit(proc)
    kwargs: dict = {
        "videos_kwargs": {
            "size": {"longest_edge": limit * 8, "shortest_edge": limit * 4},
        },
    }

    proc._clamp_dynamic_video_call_kwargs(kwargs)

    clamped = kwargs["videos_kwargs"]["size"]
    assert clamped["longest_edge"] == limit
    assert clamped["shortest_edge"] == limit


def test_video_call_kwargs_top_level_do_resize_false_hard_fails() -> None:
    """Video ``do_resize=False`` at top level must raise loudly."""
    proc = _make_processor()
    with pytest.raises(ValueError, match="do_resize=False"):
        proc._clamp_dynamic_video_call_kwargs({"do_resize": False})


def test_video_call_kwargs_nested_do_resize_false_hard_fails() -> None:
    """Video ``videos_kwargs={'do_resize': False}`` is caught the same way."""
    proc = _make_processor()
    with pytest.raises(ValueError, match="do_resize=False"):
        proc._clamp_dynamic_video_call_kwargs({"videos_kwargs": {"do_resize": False}})


def test_video_call_kwargs_missing_video_processor_is_noop() -> None:
    """The clamp must survive a bare processor with no video processor attached."""
    proc = _make_processor()
    proc.video_processor = None

    proc._clamp_dynamic_video_call_kwargs({"size": {"longest_edge": 10**9}})


# ---------------------------------------------------------------------------
# Video path — end-to-end via ``__call__``
# ---------------------------------------------------------------------------


def test_call_forwards_clamped_video_kwargs_to_super(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The video path's ``__call__`` clamps caller video kwargs before dispatch."""
    proc = _make_processor()
    limit = _expected_video_limit(proc)
    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    result = proc(
        images=None,
        text="describe <|video_pad|>",
        videos=[object()],
        videos_kwargs={"size": {"longest_edge": limit * 8, "shortest_edge": limit * 4}},
    )

    assert result == "sentinel"
    forwarded = captured["kwargs"]
    assert forwarded["videos_kwargs"]["size"]["longest_edge"] == limit
    assert forwarded["videos_kwargs"]["size"]["shortest_edge"] == limit


def test_video_do_resize_false_bypass_hard_fails_before_super_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video ``do_resize=False`` guard must trip before the heavy decode."""
    reached = {"super": False}

    def _boom_super(self, *args, **kwargs):
        reached["super"] = True
        raise AssertionError("super().__call__ must not run when do_resize=False")

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _boom_super)

    proc = _make_processor()
    with pytest.raises(ValueError, match="do_resize=False"):
        proc(
            images=None,
            text="describe <|video_pad|>",
            videos=[object()],
            videos_kwargs={"do_resize": False},
        )
    assert reached["super"] is False


# ---------------------------------------------------------------------------
# Video path — real video processor produces a per-frame grid inside the budget
# ---------------------------------------------------------------------------


def test_huge_videos_kwargs_size_still_produces_within_budget_video_grid() -> None:
    """A ``videos_kwargs={'size': {...}}`` override + 4K synthetic video stays in budget.

    Runs the real upstream video processor with the clamped size to confirm
    the produced ``video_grid_thw`` respects the per-frame token budget.
    """
    proc = _make_processor()
    limit = _expected_video_limit(proc)

    kwargs: dict = {
        "videos_kwargs": {
            "size": {"longest_edge": limit * 8, "shortest_edge": limit * 4},
        },
    }
    proc._clamp_dynamic_video_call_kwargs(kwargs)

    size = kwargs["videos_kwargs"]["size"]
    frames = torch.zeros((4, 3, 2160, 3840), dtype=torch.uint8)
    result = proc.video_processor(videos=frames, size=size, do_sample_frames=False)
    grid_thw = result["video_grid_thw"]

    assert grid_thw.ndim == 2 and grid_thw.shape[1] == 3
    per_frame_tokens = grid_thw[:, 1] * grid_thw[:, 2]
    assert bool((per_frame_tokens > 0).all())
    assert bool((per_frame_tokens <= proc.max_vision_tokens).all()), (
        f"per-frame grid exceeded budget: {per_frame_tokens.tolist()} > {proc.max_vision_tokens}"
    )


# ---------------------------------------------------------------------------
# Sanity: the clamp only fires on the dynamic-vision path
# ---------------------------------------------------------------------------


def test_static_vision_image_path_ignores_caller_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static-vision keeps forcing the fixed grid via ``_resize_images``; caller
    ``max_pixels`` / ``do_resize`` overrides must not trigger the dynamic clamp.

    The static path relies on ``_resize_images`` producing a fixed image
    resolution and on the MXQ's baked grid rejecting shape mismatches; the
    dynamic clamp isn't in the code path at all here, so caller overrides pass
    through untouched. This test pins down that we haven't accidentally
    plumbed the guard into the static branch.
    """
    proc = _make_processor(dynamic_vision=False)

    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    # ``_resize_images`` handles a bare object() poorly; hand it a PIL image.
    from PIL import Image

    image = Image.new("RGB", (224, 224))
    result = proc(
        images=[image],
        text="describe <|image_pad|>",
        do_resize=False,  # Would raise on the dynamic path; must pass on static.
    )

    assert result == "sentinel"
    assert captured["kwargs"]["do_resize"] is False
