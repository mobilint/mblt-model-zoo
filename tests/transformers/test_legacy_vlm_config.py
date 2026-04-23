"""Regression tests for legacy vision-language config loading."""

import inspect

import pytest

from mblt_model_zoo.hf_transformers.models.qwen2_vl.configuration_qwen2_vl import (
    MobilintQwen2VLConfig,
    MobilintQwen2VLTextConfig,
)
from tests.transformers.qwen3_vl_compat import transformers_supports_qwen3_vl

if transformers_supports_qwen3_vl():
    from mblt_model_zoo.hf_transformers.models.qwen3_vl.configuration_qwen3_vl import (
        MobilintQwen3VLTextConfig,
    )


TEXT_CONFIG_CASES = [pytest.param(MobilintQwen2VLTextConfig, id="qwen2_vl")]

if transformers_supports_qwen3_vl():
    TEXT_CONFIG_CASES.append(pytest.param(MobilintQwen3VLTextConfig, id="qwen3_vl"))


@pytest.mark.parametrize("text_config_cls", TEXT_CONFIG_CASES)
def test_vlm_text_config_signature_exposes_mobilint_backend_fields(text_config_cls: type) -> None:
    """Expose Mobilint backend kwargs so upstream VLM flat-config loading can discover them."""
    signature = inspect.signature(text_config_cls.__init__)

    for field_name in ("mxq_path", "dev_no", "core_mode", "target_cores", "target_clusters"):
        assert field_name in signature.parameters


def test_qwen2_vl_flat_text_backend_fields_are_routed_into_text_config() -> None:
    """Route flat legacy text backend fields through upstream Qwen2-VL text-config reconstruction."""
    config_dict = MobilintQwen2VLConfig().to_dict()
    config_dict["hidden_size"] = 1536
    config_dict["mxq_path"] = "legacy-text.mxq"
    config_dict["core_mode"] = "global4"
    config_dict["target_clusters"] = [0]
    config_dict["vision_config"]["mxq_path"] = "vision.mxq"
    config_dict["text_config"] = None

    config = MobilintQwen2VLConfig.from_dict(config_dict)

    assert config.text_mxq_path == "legacy-text.mxq"
    assert config.text_core_mode == "global4"
    assert config.text_config.hidden_size == 1536
    assert config.text_config.to_dict()["target_clusters"] == [0]
    assert config.vision_mxq_path == "vision.mxq"


def test_qwen2_vl_nested_text_backend_fields_take_precedence() -> None:
    """Keep explicit nested text backend settings when legacy top-level fields also exist."""
    config_dict = MobilintQwen2VLConfig().to_dict()
    config_dict["mxq_path"] = "legacy-text.mxq"
    config_dict["core_mode"] = "single"
    config_dict["target_cores"] = ["0:0"]
    config_dict["text_config"] = {
        **config_dict["text_config"],
        "mxq_path": "nested-text.mxq",
        "core_mode": "global8",
        "target_clusters": [0, 1],
    }

    config = MobilintQwen2VLConfig.from_dict(config_dict)

    assert config.text_mxq_path == "nested-text.mxq"
    assert config.text_core_mode == "global8"
    assert config.text_config.to_dict()["target_clusters"] == [0, 1]
