"""Regression tests: the no-override default call stays inside the NPU vision-token ceiling.

``_clamp_dynamic_image_size`` used to update only ``ip.size``. On the tf 4.x line
that ships with this repository, ``Qwen2VLImageProcessor.preprocess()`` reads a
missing ``max_pixels`` from a separate scalar attribute (``self.max_pixels``) and
its "backward-compatibility" branch derives the effective ``size`` as
``{shortest_edge: self.min_pixels, longest_edge: self.max_pixels}``, ignoring
``self.size`` entirely. That meant a caller who supplied *no* override at all
could still smuggle an oversized image past the ceiling — the storage-level
clamp did nothing on the default path and the NPU could hang on a large
enough input.

These tests pin the fix: the scalar attribute is written alongside ``size``,
and a large default call produces a grid that satisfies the token budget.
Guarded ``getattr`` keeps the branch a no-op on tf 5.x-style image processors
that dropped the scalar attribute.
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


def _make_processor_with_real_image_processor(
    dynamic_vision: bool = True,
) -> MobilintQwen3VLProcessor:
    """Build a processor around the real upstream ``Qwen2VLImageProcessor``."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.image_processor = Qwen2VLImageProcessor()
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    proc.dynamic_vision = dynamic_vision
    return proc


# ---------------------------------------------------------------------------
# Storage-level clamp: scalar `max_pixels` attribute is updated alongside `size`.
# ---------------------------------------------------------------------------


def test_clamp_writes_scalar_max_pixels_on_real_image_processor() -> None:
    """The scalar attribute the tf 4.x fallback reads must be capped too."""
    proc = _make_processor_with_real_image_processor()
    ip = proc.image_processor
    limit = proc.max_vision_tokens * ip.patch_size**2
    # Sanity: the default upstream ceiling is above the NPU budget, otherwise
    # this test would be a no-op and would not exercise the fix.
    assert ip.max_pixels > limit

    proc._clamp_dynamic_image_size()

    assert ip.max_pixels == limit
    assert ip.size["longest_edge"] == limit


def test_clamp_leaves_small_min_pixels_untouched_on_real_image_processor() -> None:
    """``min_pixels`` below the ceiling must not be inflated to the limit."""
    proc = _make_processor_with_real_image_processor()
    ip = proc.image_processor
    original_min_pixels = ip.min_pixels
    limit = proc.max_vision_tokens * ip.patch_size**2
    assert original_min_pixels <= limit

    proc._clamp_dynamic_image_size()

    assert ip.min_pixels == original_min_pixels


def test_clamp_caps_oversized_min_pixels_on_real_image_processor() -> None:
    """``min_pixels`` above the ceiling would inflate a small input past the budget."""
    proc = _make_processor_with_real_image_processor()
    ip = proc.image_processor
    limit = proc.max_vision_tokens * ip.patch_size**2
    ip.min_pixels = limit * 4  # simulate a caller-modified processor

    proc._clamp_dynamic_image_size()

    assert ip.min_pixels == limit


def test_clamp_survives_image_processor_without_scalar_attribute() -> None:
    """The scalar branch must be a no-op when the attribute is absent (tf 5.x)."""

    class _NoScalarImageProcessor:
        def __init__(self) -> None:
            self.patch_size = 14
            self.size = {"longest_edge": 4096 * 4096, "shortest_edge": 3136}

    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.image_processor = _NoScalarImageProcessor()
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    proc.dynamic_vision = True

    # Must not raise (no ``max_pixels`` attribute to touch).
    proc._clamp_dynamic_image_size()

    limit = proc.max_vision_tokens * proc.image_processor.patch_size**2
    assert proc.image_processor.size["longest_edge"] == limit
    assert not hasattr(proc.image_processor, "max_pixels")


# ---------------------------------------------------------------------------
# End-to-end: default call (no ``max_pixels`` / ``size`` override) stays in budget.
# ---------------------------------------------------------------------------


def test_default_call_on_oversized_image_stays_within_token_budget() -> None:
    """No caller overrides: a 4K image must still land inside ``max_vision_tokens``.

    This is the exact leak the reviewer called out: on tf 4.x, the effective
    ``max_pixels`` for a default call is ``self.max_pixels`` (via the
    backward-compatibility branch), not ``self.size["longest_edge"]``. Before
    the fix, ``_clamp_dynamic_image_size`` updated only ``self.size`` and
    left the scalar at the original ~1M-pixel ceiling, so a large enough
    image produced a >2048-token grid and could hang the NPU.
    """
    proc = _make_processor_with_real_image_processor()
    ip = proc.image_processor

    proc._clamp_dynamic_image_size()

    # A 4K synthetic image with no per-call size/max_pixels overrides.
    image = torch.zeros((3, 2160, 3840), dtype=torch.uint8).numpy()
    result = ip(images=[image])
    grid_thw = result["image_grid_thw"]
    per_image_tokens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]

    assert bool((per_image_tokens > 0).all())
    assert bool((per_image_tokens <= proc.max_vision_tokens).all()), (
        f"per-image grid exceeded budget: {per_image_tokens.tolist()} > {proc.max_vision_tokens}"
    )


def test_default_call_through_processor_stays_within_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end via ``__call__``: no caller overrides, storage clamp is in effect.

    Companion to the direct-processor test above — asserts the ``__call__``
    wrapper actually runs ``_clamp_dynamic_image_size`` before dispatch, so
    the default call path is fully covered. Uses a fake ``super().__call__``
    that inspects the *live* image processor state after the clamp has run
    and produces the grid via the real underlying processor.
    """
    proc = _make_processor_with_real_image_processor()
    ip = proc.image_processor
    limit = proc.max_vision_tokens * ip.patch_size**2

    seen: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        seen["max_pixels"] = self.image_processor.max_pixels
        seen["longest_edge"] = self.image_processor.size["longest_edge"]
        result = self.image_processor(images=[images[0]])
        seen["grid_thw"] = result["image_grid_thw"]
        return "sentinel"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    image = torch.zeros((3, 2160, 3840), dtype=torch.uint8).numpy()
    proc(images=[image], text="describe <|image_pad|>")

    assert seen["max_pixels"] == limit
    assert seen["longest_edge"] == limit
    grid_thw = seen["grid_thw"]
    per_image_tokens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    assert bool((per_image_tokens <= proc.max_vision_tokens).all()), (
        f"per-image grid exceeded budget after storage clamp: {per_image_tokens.tolist()} > {proc.max_vision_tokens}"
    )
