import importlib
import inspect

import pytest
import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import skip_if_transformers_lacks_qwen3_vl_support

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import MobilintQwen3VLForConditionalGeneration


@pytest.mark.parametrize(
    ("model_cls", "method_name"),
    [
        (MobilintQwen3VLForConditionalGeneration, "forward"),
        (MobilintQwen3VLForConditionalGeneration, "prepare_inputs_for_generation"),
    ],
)
def test_mobilint_qwen3_vl_generation_hooks_accept_count_npu_time(model_cls, method_name: str):
    signature = inspect.signature(getattr(model_cls, method_name))

    assert "count_npu_time" in signature.parameters


@pytest.mark.parametrize(
    "model_path",
    [
        "mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl:MobilintQwen3VLForConditionalGeneration",
        "mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl:MobilintQwen3VLTextModel",
    ],
)
def test_mobilint_qwen3_vl_generation_hooks_expose_inputs_embeds(model_path: str):
    module_name, class_name = model_path.split(":")
    model_cls = getattr(importlib.import_module(module_name), class_name)

    forward_signature = inspect.signature(model_cls.forward)
    prepare_signature = inspect.signature(model_cls.prepare_inputs_for_generation)

    assert "inputs_embeds" in forward_signature.parameters
    assert "inputs_embeds" in prepare_signature.parameters


def test_qwen3_vl_prepare_inputs_preserves_npu_prefill_chunk_size(monkeypatch: pytest.MonkeyPatch):
    signature = inspect.signature(MobilintQwen3VLForConditionalGeneration.prepare_inputs_for_generation)

    assert "npu_prefill_chunk_size" in signature.parameters

    def _base_prepare_inputs_for_generation(*args, **kwargs):
        del args, kwargs
        return {"input_ids": torch.tensor([[1]])}

    monkeypatch.setattr(
        "mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl."
        "Qwen3VLForConditionalGeneration.prepare_inputs_for_generation",
        _base_prepare_inputs_for_generation,
    )
    model = object.__new__(MobilintQwen3VLForConditionalGeneration)

    model_inputs = model.prepare_inputs_for_generation(
        torch.tensor([[1]]),
        count_npu_time=True,
        npu_prefill_chunk_size=64,
    )

    assert model_inputs["count_npu_time"] is True
    assert model_inputs["npu_prefill_chunk_size"] == 64