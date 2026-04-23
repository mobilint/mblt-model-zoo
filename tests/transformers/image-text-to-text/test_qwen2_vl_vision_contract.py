"""Regression tests for the Qwen2-VL vision output contract."""

from types import SimpleNamespace

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from mblt_model_zoo.hf_transformers.models.qwen2_vl import modeling_qwen2_vl
from mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl import (
    MobilintQwen2VisionTransformerPretrainedModel,
)


class DummyQwen2Vision:
    def __init__(self, return_dict: bool = True) -> None:
        self.config = SimpleNamespace(return_dict=return_dict)
        self.mxq_inputs = []

    def mxq_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.mxq_inputs.append(hidden_states)
        return torch.arange(64 * 8, dtype=torch.float32).view(1, 64, 8)


def test_qwen2_vl_visual_forward_returns_upstream_style_output(monkeypatch) -> None:
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


def test_qwen2_vl_visual_forward_supports_return_dict_false(monkeypatch) -> None:
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


def test_qwen2_vl_visual_forward_supports_legacy_tensor_contract(monkeypatch) -> None:
    monkeypatch.setattr(modeling_qwen2_vl, "_upstream_qwen2_vl_uses_structured_vision_outputs", lambda: False)
    dummy = DummyQwen2Vision()
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen2VisionTransformerPretrainedModel.forward(dummy, pixel_values, grid_thw)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (64, 8)
