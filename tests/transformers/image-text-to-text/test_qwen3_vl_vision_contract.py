"""Regression tests for the Qwen3-VL vision output contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mblt_model_zoo.hf_transformers.models.qwen3_vl import modeling_qwen3_vl
from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    MobilintQwen3VLModel,
    MobilintQwen3VLVisionModel,
)


class DummyVisionMxqModel:
    """Stub MXQ vision backend used by the contract tests."""

    def __init__(self) -> None:
        """Initialize the recorded input list."""
        self.inputs: list[np.ndarray] = []

    def infer(self, npu_inputs: np.ndarray) -> list[np.ndarray]:
        """Record the MXQ input and return placeholder encoder outputs."""
        self.inputs.append(npu_inputs)
        return [np.zeros((1,), dtype=np.float32)]


class DummyQwen3Vision:
    """Stub vision tower exposing the methods used by the Mobilint wrapper."""

    dtype = torch.float32
    spatial_merge_size = 2

    def __init__(self, return_dict: bool = True) -> None:
        """Initialize the dummy config and MXQ backend."""
        self.config = SimpleNamespace(return_dict=return_dict)
        self.mxq_model = DummyVisionMxqModel()
        self.call_kwargs: list[dict[str, object]] = []

    def _prepare_npu_inputs(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> np.ndarray:
        """Return the fixed-shape MXQ input expected by the runtime."""
        del hidden_states, grid_thw
        return np.zeros((1024, 64, 6), dtype=np.float32)

    def get_mxq_model(self) -> DummyVisionMxqModel:
        """Return the stub MXQ backend."""
        return self.mxq_model

    def _reorder_encoder_outputs(
        self,
        encoder_outputs: list[np.ndarray],
        device: torch.device,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return deterministic image and deepstack embeddings."""
        del encoder_outputs
        image_embeds = torch.arange(64 * 8, dtype=torch.float32, device=device).view(64, 8)
        deepstack_embeds = [
            torch.ones((64, 8), dtype=torch.float32, device=device),
            torch.full((64, 8), 2.0, dtype=torch.float32, device=device),
            torch.full((64, 8), 3.0, dtype=torch.float32, device=device),
        ]
        return image_embeds, deepstack_embeds

    def __call__(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        """Route calls through the real wrapper implementation."""
        self.call_kwargs.append(dict(kwargs))
        return MobilintQwen3VLVisionModel.forward(self, pixel_values, grid_thw, **kwargs)


class DummyQwen3VLModel:
    """Stub multimodal model exposing only the fields used by get_image_features."""

    def __init__(self, return_dict: bool = True) -> None:
        """Initialize the dummy config and vision tower."""
        self.config = SimpleNamespace(return_dict=return_dict)
        self.visual = DummyQwen3Vision(return_dict=return_dict)


def test_qwen3_vl_visual_forward_returns_upstream_style_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return structured vision outputs by default."""
    monkeypatch.setattr(modeling_qwen3_vl, "_upstream_qwen3_vl_uses_structured_vision_outputs", lambda: True)
    dummy = DummyQwen3Vision()
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen3VLVisionModel.forward(dummy, pixel_values, grid_thw)

    assert isinstance(outputs, BaseModelOutputWithDeepstackFeatures)
    assert outputs.last_hidden_state is None
    assert outputs.hidden_states is None
    assert outputs.attentions is None
    assert outputs.pooler_output.shape == (64, 8)
    assert len(outputs.deepstack_features) == 3
    assert dummy.mxq_model.inputs[0].shape == (1024, 64, 6)


def test_qwen3_vl_visual_forward_supports_return_dict_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert structured vision outputs to tuple form when requested."""
    monkeypatch.setattr(modeling_qwen3_vl, "_upstream_qwen3_vl_uses_structured_vision_outputs", lambda: True)
    dummy = DummyQwen3Vision()
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen3VLVisionModel.forward(dummy, pixel_values, grid_thw, return_dict=False)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    assert outputs[0].shape == (64, 8)
    assert len(outputs[1]) == 3


def test_qwen3_vl_visual_forward_supports_legacy_tuple_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the legacy tuple form when installed upstream still expects tuple outputs."""
    monkeypatch.setattr(modeling_qwen3_vl, "_upstream_qwen3_vl_uses_structured_vision_outputs", lambda: False)
    dummy = DummyQwen3Vision()
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen3VLVisionModel.forward(dummy, pixel_values, grid_thw)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    assert outputs[0].shape == (64, 8)
    assert len(outputs[1]) == 3


def test_qwen3_vl_visual_forward_return_dict_true_overrides_legacy_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow explicit structured outputs even on legacy upstream contracts."""
    monkeypatch.setattr(modeling_qwen3_vl, "_upstream_qwen3_vl_uses_structured_vision_outputs", lambda: False)
    dummy = DummyQwen3Vision()
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen3VLVisionModel.forward(dummy, pixel_values, grid_thw, return_dict=True)

    assert isinstance(outputs, BaseModelOutputWithDeepstackFeatures)
    assert outputs.pooler_output.shape == (64, 8)
    assert len(outputs.deepstack_features) == 3


def test_qwen3_vl_get_image_features_supports_return_dict_false() -> None:
    """Preserve the tuple contract of the upstream image feature helper."""
    if not modeling_qwen3_vl._upstream_qwen3_vl_uses_structured_vision_outputs():
        pytest.skip("Installed Transformers uses the legacy Qwen3-VL tuple contract.")

    dummy = DummyQwen3VLModel()
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen3VLModel.get_image_features(
        dummy,
        pixel_values=pixel_values,
        image_grid_thw=grid_thw,
        return_dict=False,
    )

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    assert len(outputs[0]) == 1
    assert outputs[0][0].shape == (64, 8)
    assert len(outputs[1]) == 3
    assert dummy.visual.call_kwargs == [{"return_dict": True}]


def test_qwen3_vl_get_image_features_uses_config_return_dict_default() -> None:
    """Use the model config when get_image_features omits return_dict."""
    dummy = DummyQwen3VLModel(return_dict=False)
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen3VLModel.get_image_features(
        dummy,
        pixel_values=pixel_values,
        image_grid_thw=grid_thw,
    )

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    assert len(outputs[0]) == 1
    assert outputs[0][0].shape == (64, 8)
    assert len(outputs[1]) == 3
    expected_call_kwargs = (
        [{"return_dict": True}] if modeling_qwen3_vl._upstream_qwen3_vl_uses_structured_vision_outputs() else [{}]
    )
    assert dummy.visual.call_kwargs == expected_call_kwargs
