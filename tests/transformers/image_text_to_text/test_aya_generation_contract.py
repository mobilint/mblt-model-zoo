"""Regression tests for the Aya generation input contract."""

import pytest
import torch

from mblt_model_zoo.hf_transformers.models.aya_vision.modeling_aya_vision import (
    MobilintAyaVisionForConditionalGeneration,
)
from mblt_model_zoo.hf_transformers.utils.generation_utils import MobilintGenerationMixin


def test_aya_prepare_inputs_for_generation_keeps_images_on_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep raw image inputs on the first generation step."""
    monkeypatch.setattr(
        MobilintGenerationMixin,
        "prepare_inputs_for_generation",
        lambda *args, **kwargs: {"base": True},
    )
    dummy = object.__new__(MobilintAyaVisionForConditionalGeneration)
    pixel_values = torch.ones((1, 3, 4, 4), dtype=torch.float32)

    model_inputs = MobilintAyaVisionForConditionalGeneration.prepare_inputs_for_generation(
        dummy,
        input_ids=torch.tensor([[1]], dtype=torch.long),
        pixel_values=pixel_values,
        is_first_iteration=True,
        use_cache=True,
    )

    assert model_inputs["base"] is True
    assert model_inputs["pixel_values"] is pixel_values


def test_aya_prepare_inputs_for_generation_drops_images_for_cached_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid re-sending raw images once cached decoding has started."""
    monkeypatch.setattr(
        MobilintGenerationMixin,
        "prepare_inputs_for_generation",
        lambda *args, **kwargs: {"base": True},
    )
    dummy = object.__new__(MobilintAyaVisionForConditionalGeneration)
    pixel_values = torch.ones((1, 3, 4, 4), dtype=torch.float32)

    model_inputs = MobilintAyaVisionForConditionalGeneration.prepare_inputs_for_generation(
        dummy,
        input_ids=torch.tensor([[1]], dtype=torch.long),
        pixel_values=pixel_values,
        cache_position=torch.tensor([1], dtype=torch.long),
        is_first_iteration=False,
        use_cache=True,
    )

    assert model_inputs == {"base": True}


def test_aya_prepare_inputs_for_generation_keeps_images_on_cache_position_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep raw image inputs when cache position marks the first prefill step."""
    monkeypatch.setattr(
        MobilintGenerationMixin,
        "prepare_inputs_for_generation",
        lambda *args, **kwargs: {"base": True},
    )
    dummy = object.__new__(MobilintAyaVisionForConditionalGeneration)
    pixel_values = torch.ones((1, 3, 4, 4), dtype=torch.float32)

    model_inputs = MobilintAyaVisionForConditionalGeneration.prepare_inputs_for_generation(
        dummy,
        input_ids=torch.tensor([[1]], dtype=torch.long),
        pixel_values=pixel_values,
        cache_position=torch.tensor([0], dtype=torch.long),
        is_first_iteration=False,
        use_cache=True,
    )

    assert model_inputs["base"] is True
    assert model_inputs["pixel_values"] is pixel_values


def test_aya_prepare_inputs_for_generation_keeps_images_without_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep raw image inputs when generation explicitly disables cache reuse."""
    monkeypatch.setattr(
        MobilintGenerationMixin,
        "prepare_inputs_for_generation",
        lambda *args, **kwargs: {"base": True},
    )
    dummy = object.__new__(MobilintAyaVisionForConditionalGeneration)
    pixel_values = torch.ones((1, 3, 4, 4), dtype=torch.float32)

    model_inputs = MobilintAyaVisionForConditionalGeneration.prepare_inputs_for_generation(
        dummy,
        input_ids=torch.tensor([[1]], dtype=torch.long),
        pixel_values=pixel_values,
        is_first_iteration=False,
        use_cache=False,
    )

    assert model_inputs["base"] is True
    assert model_inputs["pixel_values"] is pixel_values
