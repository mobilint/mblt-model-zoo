try:
    from mblt_model_zoo.hf_transformers.models.qwen3.configuration_qwen3 import (
        MobilintQwen3Config,
    )
    from mblt_model_zoo.hf_transformers.models.qwen3.modeling_qwen3 import (
        MobilintQwen3ForCausalLM,
    )
except ImportError:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        "Please run: pip install mblt_model_zoo[transformers]"
    )

__all__ = ["MobilintQwen3Config", "MobilintQwen3ForCausalLM"]