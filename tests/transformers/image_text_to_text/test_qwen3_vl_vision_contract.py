"""Regression tests for the Qwen3-VL vision output contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import skip_if_transformers_lacks_qwen3_vl_support

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl import modeling_qwen3_vl  # noqa: E402
from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
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
        batch_size = int(npu_inputs.shape[0]) if npu_inputs.ndim == 4 else 1
        values = [1.0, 3.0, 4.0, 2.0]
        return [np.full((batch_size, 64, 8), value, dtype=np.float32) for value in values]


class DummyQwen3Vision:
    """Stub vision tower exposing the methods used by the Mobilint wrapper."""

    dtype = torch.float32
    spatial_merge_size = 2

    def __init__(self, return_dict: bool = True, core_mode: str = "single") -> None:
        """Initialize the dummy config and MXQ backend."""
        self.config = SimpleNamespace(return_dict=return_dict, core_mode=core_mode)
        self.mxq_model = DummyVisionMxqModel()
        self.call_kwargs: list[dict[str, object]] = []

    def _prepare_npu_inputs(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> np.ndarray:
        """Return the fixed-shape MXQ input expected by the runtime."""
        del hidden_states, grid_thw
        return np.zeros((1024, 64, 6), dtype=np.float32)

    def _split_hidden_states_by_grid(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Delegate grid splitting to the wrapper implementation."""
        return MobilintQwen3VLVisionModel._split_hidden_states_by_grid(self, hidden_states, grid_thw)

    def get_mxq_model(self) -> DummyVisionMxqModel:
        """Return the stub MXQ backend."""
        return self.mxq_model

    def _reorder_encoder_outputs(
        self,
        encoder_outputs: list[np.ndarray],
        device: torch.device,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return deterministic image and deepstack embeddings."""
        del encoder_outputs
        image_embeds = torch.arange(batch_size * 64 * 8, dtype=torch.float32, device=device).view(batch_size * 64, 8)
        deepstack_embeds = [
            torch.ones((batch_size * 64, 8), dtype=torch.float32, device=device),
            torch.full((batch_size * 64, 8), 2.0, dtype=torch.float32, device=device),
            torch.full((batch_size * 64, 8), 3.0, dtype=torch.float32, device=device),
        ]
        return image_embeds, deepstack_embeds

    def _encode_images(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Delegate image encoding to the wrapper implementation."""
        return MobilintQwen3VLVisionModel._encode_images(self, hidden_states, grid_thw)

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


def test_qwen3_vl_visual_forward_uses_config_return_dict_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the vision config default when structured upstream omits ``return_dict``."""
    monkeypatch.setattr(modeling_qwen3_vl, "_upstream_qwen3_vl_uses_structured_vision_outputs", lambda: True)
    dummy = DummyQwen3Vision(return_dict=False)
    pixel_values = torch.zeros((256, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen3VLVisionModel.forward(dummy, pixel_values, grid_thw)

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


@pytest.mark.parametrize("core_mode", ["single", "global4", "global8"])
def test_qwen3_vl_visual_forward_loops_batched_images_for_non_multi_core_modes(
    monkeypatch: pytest.MonkeyPatch,
    core_mode: str,
) -> None:
    """Run one MXQ call per image when the core mode does not support batched vision inputs."""
    monkeypatch.setattr(modeling_qwen3_vl, "_upstream_qwen3_vl_uses_structured_vision_outputs", lambda: True)
    dummy = DummyQwen3Vision(core_mode=core_mode)
    pixel_values = torch.zeros((512, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16], [1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen3VLVisionModel.forward(dummy, pixel_values, grid_thw)

    assert isinstance(outputs, BaseModelOutputWithDeepstackFeatures)
    assert outputs.pooler_output.shape == (128, 8)
    assert len(outputs.deepstack_features) == 3
    assert [feature.shape for feature in outputs.deepstack_features] == [torch.Size([128, 8])] * 3
    assert len(dummy.mxq_model.inputs) == 2
    assert [item.shape for item in dummy.mxq_model.inputs] == [(1024, 64, 6), (1024, 64, 6)]


def test_qwen3_vl_visual_forward_uses_batched_input_for_multi_core_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use one batched MXQ call for Qwen3-VL vision inputs in multi core mode."""
    monkeypatch.setattr(modeling_qwen3_vl, "_upstream_qwen3_vl_uses_structured_vision_outputs", lambda: True)
    dummy = DummyQwen3Vision(core_mode="multi")
    pixel_values = torch.zeros((512, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 16, 16], [1, 16, 16]], dtype=torch.long)

    outputs = MobilintQwen3VLVisionModel.forward(dummy, pixel_values, grid_thw)

    assert isinstance(outputs, BaseModelOutputWithDeepstackFeatures)
    assert outputs.pooler_output.shape == (128, 8)
    assert len(outputs.deepstack_features) == 3
    assert [feature.shape for feature in outputs.deepstack_features] == [torch.Size([128, 8])] * 3
    assert len(dummy.mxq_model.inputs) == 1
    assert dummy.mxq_model.inputs[0].shape == (2, 1024, 64, 6)
