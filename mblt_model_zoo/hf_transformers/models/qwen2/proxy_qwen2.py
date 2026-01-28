try:
    from mblt_model_zoo.hf_transformers.models.qwen2.configuration_qwen2 import (
        MobilintQwen2Config,
    )
    from mblt_model_zoo.hf_transformers.models.qwen2.modeling_qwen2 import (
        MobilintQwen2ForCausalLM,
    )
except ImportError:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        "Please run: pip install mblt_model_zoo[transformers]"
    )

__all__ = ["MobilintQwen2Config", "MobilintQwen2ForCausalLM"]