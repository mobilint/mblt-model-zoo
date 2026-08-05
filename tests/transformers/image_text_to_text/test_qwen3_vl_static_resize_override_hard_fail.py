"""Regression tests: static-vision Qwen3-VL rejects caller-supplied resize overrides.

The static-vision release ships a vision MXQ compiled for a rigid
``(H, W, C) = (1024, 64, 6)`` grid derived from the shipped ``patch_size`` /
``merge_size``. ``MobilintQwen3VLProcessor.__call__`` pre-resizes every input
to the matching pixel resolution via ``_resize_images``, but the upstream
image processor's ``_preprocess`` still runs after that step and honors
caller-supplied ``size`` / ``min_pixels`` / ``max_pixels`` / ``do_resize=False``
overrides — which re-resize the just-normalized image away from the grid the
compiled MXQ expects. Concrete failure: a large ``min_pixels`` inflates the
pre-resized 224x224 image, patch extraction produces a grid that no longer
matches ``(1024, 64, 6)``, and the MXQ either shape-mismatches or silently
produces semantically wrong output.

These tests pin down the processor-level guard on both the top-level and
nested ``images_kwargs`` scopes, verify the reject fires before
``super().__call__``, and confirm the dynamic path still accepts the same
overrides (its clamp path is owned by the dynamic-vision guard tests).
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


# ---------------------------------------------------------------------------
# Static branch — direct method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_pixels", 1_000_000),
        ("max_pixels", 1_000_000),
        ("size", {"longest_edge": 4096, "shortest_edge": 4096}),
    ],
)
def test_static_reject_top_level_resize_override_raises(field: str, value) -> None:
    """A top-level ``size`` / ``min_pixels`` / ``max_pixels`` override raises."""
    proc = _make_processor(dynamic_vision=False)
    with pytest.raises(ValueError, match="static-vision"):
        proc._reject_static_image_resize_overrides({field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_pixels", 1_000_000),
        ("max_pixels", 1_000_000),
        ("size", {"longest_edge": 4096, "shortest_edge": 4096}),
    ],
)
def test_static_reject_nested_images_kwargs_resize_override_raises(field: str, value) -> None:
    """A ``images_kwargs={<field>: ...}`` override is caught the same way."""
    proc = _make_processor(dynamic_vision=False)
    with pytest.raises(ValueError, match="static-vision"):
        proc._reject_static_image_resize_overrides({"images_kwargs": {field: value}})


def test_static_reject_top_level_do_resize_false_raises() -> None:
    """``do_resize=False`` bypasses the pre-resize contract and raises."""
    proc = _make_processor(dynamic_vision=False)
    with pytest.raises(ValueError, match="static-vision"):
        proc._reject_static_image_resize_overrides({"do_resize": False})


def test_static_reject_nested_do_resize_false_raises() -> None:
    """``images_kwargs={'do_resize': False}`` is caught in the nested dict too."""
    proc = _make_processor(dynamic_vision=False)
    with pytest.raises(ValueError, match="static-vision"):
        proc._reject_static_image_resize_overrides({"images_kwargs": {"do_resize": False}})


def test_static_reject_do_resize_true_is_noop() -> None:
    """``do_resize=True`` matches the branch's assumption and does not raise."""
    proc = _make_processor(dynamic_vision=False)
    # Should not raise.
    proc._reject_static_image_resize_overrides({"do_resize": True})


def test_static_reject_none_valued_override_is_noop() -> None:
    """Fields present with a ``None`` value are treated as no override."""
    proc = _make_processor(dynamic_vision=False)
    # Should not raise for either top-level or nested None values.
    proc._reject_static_image_resize_overrides({"size": None, "images_kwargs": {"max_pixels": None}})


def test_static_reject_no_overrides_is_noop() -> None:
    """An empty kwargs dict is a no-op regression baseline."""
    proc = _make_processor(dynamic_vision=False)
    # Should not raise.
    proc._reject_static_image_resize_overrides({})


# ---------------------------------------------------------------------------
# Static branch — end-to-end via ``__call__``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_pixels", 1_000_000),
        ("max_pixels", 1_000_000),
        ("size", {"longest_edge": 4096, "shortest_edge": 4096}),
        ("do_resize", False),
    ],
)
def test_call_static_resize_override_hard_fails_before_super_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value,
) -> None:
    """The static reject must trip before ``super().__call__`` is entered."""
    reached = {"super": False}

    def _boom_super(self, *args, **kwargs):
        reached["super"] = True
        raise AssertionError("super().__call__ must not run when a static resize override is set")

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _boom_super)

    proc = _make_processor(dynamic_vision=False)
    with pytest.raises(ValueError, match="static-vision"):
        proc(
            images=[_make_image()],
            text="describe <|image_pad|>",
            **{field: value},
        )
    assert reached["super"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_pixels", 1_000_000),
        ("max_pixels", 1_000_000),
        ("size", {"longest_edge": 4096, "shortest_edge": 4096}),
        ("do_resize", False),
    ],
)
def test_call_static_resize_override_via_nested_scope_hard_fails(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value,
) -> None:
    """The nested ``images_kwargs`` scope is also caught by the end-to-end guard."""
    reached = {"super": False}

    def _boom_super(self, *args, **kwargs):
        reached["super"] = True
        raise AssertionError("super().__call__ must not run when a nested static resize override is set")

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _boom_super)

    proc = _make_processor(dynamic_vision=False)
    with pytest.raises(ValueError, match="static-vision"):
        proc(
            images=[_make_image()],
            text="describe <|image_pad|>",
            images_kwargs={field: value},
        )
    assert reached["super"] is False


def test_call_static_no_overrides_forwards_to_super(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No overrides → static path forwards to ``super().__call__`` (regression baseline)."""
    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    proc = _make_processor(dynamic_vision=False)
    result = proc(images=[_make_image()], text="describe <|image_pad|>")

    assert result == "sentinel"
    # Baseline: none of the resize-shaping keys leak into the upstream call.
    forwarded = captured["kwargs"]
    for field in ("size", "min_pixels", "max_pixels", "do_resize"):
        assert field not in forwarded


def test_call_static_do_resize_true_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``do_resize=True`` matches the default assumption and is forwarded."""
    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    proc = _make_processor(dynamic_vision=False)
    result = proc(
        images=[_make_image()],
        text="describe <|image_pad|>",
        do_resize=True,
    )

    assert result == "sentinel"
    assert captured["kwargs"]["do_resize"] is True


# ---------------------------------------------------------------------------
# Dynamic branch still accepts the same overrides — behavior owned by the
# dynamic clamp guard tests, exercised here to prevent cross-branch regressions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_pixels", 1_000_000),
        ("max_pixels", 1_000_000),
        ("size", {"longest_edge": 4096, "shortest_edge": 4096}),
    ],
)
def test_call_dynamic_resize_override_still_accepted(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value,
) -> None:
    """Dynamic-mode processor still forwards resize overrides through its clamp path.

    The static reject must not fire on the dynamic branch. The clamp itself is
    stubbed here (its behavior is owned by ``test_qwen3_vl_call_kwargs_clamp.py``);
    we only need to prove the dynamic branch is reached and the override arrives
    at ``super().__call__``.
    """
    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    def _noop_clamp(self):
        return None

    def _noop_clamp_kwargs(self, kwargs):
        return None

    monkeypatch.setattr(MobilintQwen3VLProcessor, "_clamp_dynamic_image_size", _noop_clamp)
    monkeypatch.setattr(
        MobilintQwen3VLProcessor,
        "_clamp_dynamic_image_call_kwargs",
        _noop_clamp_kwargs,
    )

    proc = _make_processor(dynamic_vision=True)
    result = proc(
        images=[_make_image()],
        text="describe <|image_pad|>",
        **{field: value},
    )

    assert result == "sentinel"
    assert captured["kwargs"][field] == value


def test_call_dynamic_do_resize_false_reaches_dynamic_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``do_resize=False`` on dynamic mode is handled by the dynamic clamp, not the static reject.

    The dynamic clamp raises with its own ``do_resize=False`` message pointing
    at the NPU vision-token ceiling — a different string from the static
    reject. Assert the raise comes from the dynamic path.
    """
    proc = _make_processor(dynamic_vision=True)

    # The real dynamic clamp needs ``image_processor`` to compute the ceiling;
    # stub it with a raise that mimics the real message so we can distinguish.
    def _dynamic_do_resize_false(self, kwargs):
        raise ValueError("do_resize=False bypasses the NPU vision-token ceiling for image inputs.")

    monkeypatch.setattr(MobilintQwen3VLProcessor, "_clamp_dynamic_image_size", lambda self: None)
    monkeypatch.setattr(
        MobilintQwen3VLProcessor,
        "_clamp_dynamic_image_call_kwargs",
        _dynamic_do_resize_false,
    )

    with pytest.raises(ValueError, match="NPU vision-token ceiling"):
        proc(
            images=[_make_image()],
            text="describe <|image_pad|>",
            do_resize=False,
        )
