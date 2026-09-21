"""Regression tests: dynamic-vision Qwen3-VL clamps video frames to the NPU token budget.

The dynamic vision MXQ hangs (watchdog timeout -> ``Model_NotAlive``) rather than
erroring cleanly when a frame produces more than ``max_vision_tokens`` pre-merge
patch tokens. The image path handles this via ``_clamp_dynamic_image_size``. These
tests pin down the analogous clamp for the video path so a high-resolution video
frame cannot silently overshoot the budget and hang the NPU.
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
    _update_size,
)


def _make_processor(dynamic_vision: bool) -> MobilintQwen3VLProcessor:
    """Build a bare processor without going through the heavy ``__init__``."""
    proc = object.__new__(MobilintQwen3VLProcessor)
    proc.video_processor = MobilintQwen3VLVideoProcessor()
    proc.dynamic_vision = dynamic_vision
    return proc


def _expected_video_limit(proc: MobilintQwen3VLProcessor) -> int:
    """The clamp target: `t_bar * h_bar * w_bar <= max_pixels` and `t_bar >= temporal_patch_size`."""
    vp = proc.video_processor
    return proc.max_vision_tokens * vp.patch_size**2 * vp.temporal_patch_size


def test_clamp_reduces_oversized_video_longest_edge() -> None:
    """Video processor `longest_edge` above the safe limit is capped in place."""
    proc = _make_processor(dynamic_vision=True)
    limit = _expected_video_limit(proc)
    proc.video_processor.size = _update_size(proc.video_processor.size, longest_edge=limit * 8)

    proc._clamp_dynamic_video_size()

    assert proc.video_processor.size["longest_edge"] == limit
    assert proc.video_processor.size["shortest_edge"] <= limit


def test_clamp_leaves_safe_video_longest_edge_untouched() -> None:
    """Sizes already inside the budget are not modified."""
    proc = _make_processor(dynamic_vision=True)
    original = dict(proc.video_processor.size)

    proc._clamp_dynamic_video_size()

    assert proc.video_processor.size["longest_edge"] == original["longest_edge"]
    assert proc.video_processor.size["shortest_edge"] == original["shortest_edge"]


def test_dynamic_vision_call_clamps_video_size_before_super_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`__call__` must clamp the video processor size before entering `super().__call__`."""
    proc = _make_processor(dynamic_vision=True)
    limit = _expected_video_limit(proc)
    proc.video_processor.size = _update_size(proc.video_processor.size, longest_edge=limit * 8)

    captured: dict[str, object] = {}

    def _capture_super(self, images, text, videos, **kwargs):
        captured["longest_edge"] = self.video_processor.size["longest_edge"]
        return "sentinel-batch-feature"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super)

    proc(images=None, text="describe <|video_pad|>", videos=[object()])

    assert captured["longest_edge"] == limit


def test_preprocessed_4k_video_grid_stays_within_token_budget() -> None:
    """A 4K synthetic video preprocessed after the clamp keeps per-frame `grid_h * grid_w` <= budget.

    Exercises the actual preprocessing path — the upstream ``smart_resize`` runs on the clamped
    ``longest_edge``, so the produced ``video_grid_thw`` must satisfy the per-frame token bound.
    """
    proc = _make_processor(dynamic_vision=True)
    limit = _expected_video_limit(proc)
    proc.video_processor.size = _update_size(proc.video_processor.size, longest_edge=limit * 8)

    proc._clamp_dynamic_video_size()

    frames = torch.zeros((4, 3, 2160, 3840), dtype=torch.uint8)
    result = proc.video_processor(videos=frames, do_sample_frames=False)
    grid_thw = result["video_grid_thw"]

    assert grid_thw.ndim == 2 and grid_thw.shape[1] == 3
    per_frame_tokens = grid_thw[:, 1] * grid_thw[:, 2]
    assert torch.all(per_frame_tokens > 0)
    assert torch.all(per_frame_tokens <= proc.max_vision_tokens), (
        f"per-frame grid exceeded budget: {per_frame_tokens.tolist()} > {proc.max_vision_tokens}"
    )
