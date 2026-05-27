"""Regression tests for the Qwen2-VL vision output contract."""

from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from mblt_model_zoo.hf_transformers.models.qwen2_vl import modeling_qwen2_vl
from mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl import (
    MobilintQwen2VisionTransformerPretrainedModel,
)


class DummyQwen2Vision:
    """Stub vision tower exposing the methods used by the Mobilint wrapper."""

    def __init__(self, return_dict: bool = True, core_mode: str = "single") -> None:
        """Initialize the dummy config and recorded MXQ inputs."""
        self.config = SimpleNamespace(return_dict=return_dict, core_mode=core_mode)
        self.mxq_inputs = []

    def mxq_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Record the MXQ input and return placeholder image embeddings."""
        self.mxq_inputs.append(hidden_states)
        if hidden_states.ndim == 4:
            batch_size = int(hidden_states.shape[0])
            return torch.arange(batch_size * 64 * 8, dtype=torch.float32).view(batch_size, 64, 8)
        return torch.arange(64 * 8, dtype=torch.float32).view(1, 64, 8)

    def _preprocess_image_tokens(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Delegate preprocessing to the wrapper implementation."""
        return MobilintQwen2VisionTransformerPretrainedModel._preprocess_image_tokens(self, hidden_states, grid_thw)

    def _split_hidden_states_by_grid(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Delegate grid splitting to the wrapper implementation."""
        return MobilintQwen2VisionTransformerPretrainedModel._split_hidden_states_by_grid(self, hidden_states, grid_thw)

    def _flatten_encoder_output(self, output: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        """Delegate output flattening to the wrapper implementation."""
        return MobilintQwen2VisionTransformerPretrainedModel._flatten_encoder_output(
            self,
            output,
            batch_size=batch_size,
        )

    def _encode_images(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Delegate image encoding to the wrapper implementation."""
        return MobilintQwen2VisionTransformerPretrainedModel._encode_images(self, hidden_states, grid_thw)


def test_qwen2_vl_visual_forward_returns_upstream_style_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return structured vision outputs when the upstream contract expects them."""
    monkeypatch.setattr(modeling_qwen2_vl, "_upstream_qwen2_vl_uses_structured_vision_outputs", lambda: True)
    dummy = DummyQwen2Vision()
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen2VisionTransformerPretrainedModel.forward(dummy, pixel_values, grid_thw)

    assert isinstance(outputs, BaseModelOutputWithPooling)
    assert outputs.last_hidden_state is None
    assert outputs.hidden_states is None
    assert outputs.attentions is None
    assert outputs.pooler_output.shape == (64, 8)
    assert dummy.mxq_inputs[0].shape == (1024, 64, 6)


def test_qwen2_vl_visual_forward_supports_return_dict_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert structured vision outputs to tuple form when requested."""
    monkeypatch.setattr(modeling_qwen2_vl, "_upstream_qwen2_vl_uses_structured_vision_outputs", lambda: True)
    dummy = DummyQwen2Vision()
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen2VisionTransformerPretrainedModel.forward(
        dummy,
        pixel_values,
        grid_thw,
        return_dict=False,
    )

    assert isinstance(outputs, tuple)
    assert len(outputs) == 1
    assert outputs[0].shape == (64, 8)


def test_qwen2_vl_visual_forward_uses_config_return_dict_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the vision config default when structured upstream omits ``return_dict``."""
    monkeypatch.setattr(modeling_qwen2_vl, "_upstream_qwen2_vl_uses_structured_vision_outputs", lambda: True)
    dummy = DummyQwen2Vision(return_dict=False)
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen2VisionTransformerPretrainedModel.forward(dummy, pixel_values, grid_thw)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 1
    assert outputs[0].shape == (64, 8)


def test_qwen2_vl_visual_forward_supports_legacy_tensor_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the legacy raw-tensor contract for older upstream helpers."""
    monkeypatch.setattr(modeling_qwen2_vl, "_upstream_qwen2_vl_uses_structured_vision_outputs", lambda: False)
    dummy = DummyQwen2Vision()
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen2VisionTransformerPretrainedModel.forward(dummy, pixel_values, grid_thw)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (64, 8)


@pytest.mark.parametrize("core_mode", ["single", "global4", "global8"])
def test_qwen2_vl_visual_forward_loops_batched_images_for_non_multi_core_modes(
    monkeypatch: pytest.MonkeyPatch,
    core_mode: str,
) -> None:
    """Run one MXQ call per image when the core mode does not support batched vision inputs."""
    monkeypatch.setattr(modeling_qwen2_vl, "_upstream_qwen2_vl_uses_structured_vision_outputs", lambda: True)
    dummy = DummyQwen2Vision(core_mode=core_mode)
    pixel_values = torch.zeros((512, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16], [1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen2VisionTransformerPretrainedModel.forward(dummy, pixel_values, grid_thw)

    assert isinstance(outputs, BaseModelOutputWithPooling)
    assert outputs.pooler_output.shape == (128, 8)
    assert len(dummy.mxq_inputs) == 2
    assert [tuple(item.shape) for item in dummy.mxq_inputs] == [(1024, 64, 6), (1024, 64, 6)]


def test_qwen2_vl_visual_forward_uses_batched_input_for_multi_core_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use one batched MXQ call for Qwen2-VL vision inputs in multi core mode."""
    monkeypatch.setattr(modeling_qwen2_vl, "_upstream_qwen2_vl_uses_structured_vision_outputs", lambda: True)
    dummy = DummyQwen2Vision(core_mode="multi")
    pixel_values = torch.zeros((512, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16], [1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen2VisionTransformerPretrainedModel.forward(dummy, pixel_values, grid_thw)

    assert isinstance(outputs, BaseModelOutputWithPooling)
    assert outputs.pooler_output.shape == (128, 8)
    assert len(dummy.mxq_inputs) == 1
    assert dummy.mxq_inputs[0].shape == (2, 1024, 64, 6)
