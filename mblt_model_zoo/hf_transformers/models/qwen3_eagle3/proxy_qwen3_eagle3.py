try:
    from mblt_model_zoo.hf_transformers.models.qwen3_eagle3.configuration_qwen3_eagle3 import (
        MobilintQwen3Eagle3Config,
    )
    from mblt_model_zoo.hf_transformers.models.qwen3_eagle3.modeling_qwen3_eagle3 import (
        MobilintQwen3Eagle3ForCausalLM,
    )
except ImportError as e:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        "Please run: pip install mblt_model_zoo[transformers]"
    ) from e


__all__ = ["MobilintQwen3Eagle3Config", "MobilintQwen3Eagle3ForCausalLM"]
