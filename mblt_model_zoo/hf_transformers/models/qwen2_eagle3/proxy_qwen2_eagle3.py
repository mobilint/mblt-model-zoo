try:
    from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.configuration_qwen2_eagle3 import (
        MobilintQwen2Eagle3Config,
    )
    from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
        MobilintQwen2Eagle3ForCausalLM,
    )
except ImportError as e:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        "Please run: pip install mblt_model_zoo[transformers]"
    ) from e


__all__ = ["MobilintQwen2Eagle3Config", "MobilintQwen2Eagle3ForCausalLM"]
