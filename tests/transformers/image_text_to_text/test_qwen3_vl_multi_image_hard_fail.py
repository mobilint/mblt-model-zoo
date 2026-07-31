"""Regression tests: static-vision Qwen3-VL rejects multi-image inputs with a clear error.

Static Qwen3-VL MXQ releases (single-input vision, fixed visual-token count in the
text decoder) bake a single image's 2D RoPE grid into the text decoder. A second
image would need its own independent 2D coordinates, which the baked rope cannot
express, so the decoder silently loses the image-boundary distinction and the
language model emits grammatically-plausible but semantically wrong output. These
tests pin down both the processor-level guard (which also skips the image
processor's patch extraction) and the model-level defense-in-depth guard, and
verify that the single-image path is unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

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


def test_encode_images_rejects_multi_image_grid_on_static_vision() -> None:
    """Bypassing the processor and driving ``_encode_images`` directly still fails.

    Defense in depth for callers that construct pixel tensors + grids by hand
    (e.g. custom pipelines, test harnesses) and skip the processor's guard.
    """
    dummy = _StaticVisionStub()
    # Two image grids (gt=1 each, distinct 2D shapes) — with gh=gw=1 the shape
    # math is trivial and we only need the number of grid rows to be > 1 to
    # trip the multi-image guard.
    grid_thw = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
    hidden_states = torch.zeros((2, 8), dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="Multi-image input requires"):
        MobilintQwen3VLVisionModel._encode_images(dummy, hidden_states, grid_thw)


def test_encode_images_allows_single_image_grid_on_static_vision() -> None:
    """Sanity check: len(grid_thw)==1 image grid does not trip the multi-image guard."""

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
