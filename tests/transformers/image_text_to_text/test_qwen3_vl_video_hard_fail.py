"""Regression tests: static-vision Qwen3-VL rejects video inputs with a clear error.

Static Qwen3-VL MXQ releases (1-input vision, fixed visual-token count in the text
decoder) cannot express video: per-frame RoPE and variable-length visual regions
are exactly what the dynamic vision MXQ was compiled to carry. Prior to the hard
fail the processor silently ran each frame through the static path and the
language model emitted grammatically-plausible but semantically empty output.
These tests pin down both the processor-level guard (which also skips the heavy
torchcodec/FFmpeg video decode) and the model-level defense-in-depth guard.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # noqa: E402

from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLVisionModel,
)
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


def test_processor_rejects_video_when_static_vision() -> None:
    """Static-mode processor must raise ``NotImplementedError`` on any video input."""
    proc = _make_processor(dynamic_vision=False)
    with pytest.raises(NotImplementedError, match="dynamic-vision Qwen3-VL release"):
        proc(images=None, text="dummy prompt", videos=[object()])


def test_processor_video_guard_fires_before_video_processor_touches_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hard fail must trip before the video processor tries to decode the input.

    Silent-fail regression was that the processor happily ran the whole
    decode pipeline (torchcodec + FFmpeg) and only produced garbage at
    inference time. Guarantee the raise happens before ``super().__call__``
    is entered — an exploding stub for the upstream call verifies we never
    reach it in the static-vision path.
    """
    called = {"super": False}

    def _boom_super_call(self, *args, **kwargs):
        called["super"] = True
        raise AssertionError("super().__call__ must not be reached in the static-video path")

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _boom_super_call)

    proc = _make_processor(dynamic_vision=False)
    with pytest.raises(NotImplementedError):
        proc(images=None, text="dummy prompt", videos=[object()])
    assert called["super"] is False


def test_processor_lets_video_through_when_dynamic_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic-mode processor keeps forwarding video to the upstream ``__call__``."""
    captured: dict[str, object] = {}

    def _capture_super_call(self, images, text, videos, **kwargs):
        captured["images"] = images
        captured["text"] = text
        captured["videos"] = videos
        return "sentinel-batch-feature"

    monkeypatch.setattr(Qwen3VLProcessor, "__call__", _capture_super_call)

    proc = _make_processor(dynamic_vision=True)
    videos_marker = [object()]
    result = proc(images=None, text="describe <|video_pad|>", videos=videos_marker)

    assert result == "sentinel-batch-feature"
    assert captured["videos"] is videos_marker


class _StaticVisionStub:
    """Minimal stand-in for ``MobilintQwen3VLVisionModel`` used by ``_encode_images``."""

    _uses_dynamic_vision = False

    def _split_hidden_states_by_grid(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        return MobilintQwen3VLVisionModel._split_hidden_states_by_grid(
            self, hidden_states, grid_thw
        )


def test_encode_images_rejects_video_grid_on_static_vision() -> None:
    """Bypassing the processor and driving ``_encode_images`` directly still fails.

    Defense in depth for callers that construct pixel tensors + grids by hand
    (e.g. custom pipelines, test harnesses) and skip the processor's guard.
    """
    dummy = _StaticVisionStub()
    # gt=4 marks a video grid (>1 frame); with gh=gw=1 the shape math is
    # trivial and we only need it to be internally consistent so
    # `_split_hidden_states_by_grid` succeeds before the guard checks `gt`.
    grid_thw = torch.tensor([[4, 1, 1]], dtype=torch.long)
    hidden_states = torch.zeros((4, 8), dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="dynamic-vision Qwen3-VL MXQ"):
        MobilintQwen3VLVisionModel._encode_images(dummy, hidden_states, grid_thw)


def test_encode_images_allows_image_grid_on_static_vision() -> None:
    """Sanity check: gt=1 image grid does not trip the video guard."""

    class _StubWithBackend(_StaticVisionStub):
        """Extend the stub with the minimum surface ``_encode_images`` touches for an image."""

        npu_backend = None

        def __init__(self) -> None:
            self.config = type("_Cfg", (), {"core_mode": "single"})()
            self.mxq_inputs: list = []

        def _prepare_npu_inputs(self, chunk: torch.Tensor, grid: torch.Tensor) -> np.ndarray:
            del chunk, grid
            return np.zeros((1024, 64, 6), dtype=np.float32)

        def get_mxq_model(self):
            outer = self

            class _MxqStub:
                def infer(self_inner, npu_input):
                    outer.mxq_inputs.append(npu_input)
                    # Match `_reorder_encoder_outputs` (4 tensors) with a 1-token grid.
                    return [np.zeros((1, 8), dtype=np.float32) for _ in range(4)]

            return _MxqStub()

        def _reorder_encoder_outputs(
            self,
            encoder_outputs,
            device: torch.device,
            batch_size: int = 1,
        ):
            del encoder_outputs, batch_size
            image_embed = torch.zeros((1, 8), dtype=torch.float32, device=device)
            deepstack_embeds = [torch.zeros((1, 8), dtype=torch.float32, device=device) for _ in range(3)]
            return image_embed, deepstack_embeds

    dummy = _StubWithBackend()
    grid_thw = torch.tensor([[1, 1, 1]], dtype=torch.long)
    hidden_states = torch.zeros((1, 8), dtype=torch.float32)

    image_embeds, deepstack = MobilintQwen3VLVisionModel._encode_images(
        dummy, hidden_states, grid_thw
    )
    assert image_embeds.shape == (1, 8)
    assert len(deepstack) == 3
    assert len(dummy.mxq_inputs) == 1
