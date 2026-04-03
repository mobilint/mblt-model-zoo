"""Proxy imports for Mobilint Qwen3-VL."""

try:
    from mblt_model_zoo.hf_transformers.models.qwen3_vl.configuration_qwen3_vl import (
        MobilintQwen3VLConfig,
    )
    from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (
        MobilintQwen3VLForConditionalGeneration,
    )
except ImportError:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. Please run: pip install mblt_model_zoo[transformers]"
    )

__all__ = ["MobilintQwen3VLConfig", "MobilintQwen3VLForConditionalGeneration"]
