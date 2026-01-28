try:
    from mblt_model_zoo.hf_transformers.models.llama.configuration_llama import (
        MobilintLlamaConfig,
    )
    from mblt_model_zoo.hf_transformers.models.llama.modeling_llama import (
        MobilintLlamaForCausalLM,
    )
except ImportError:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        "Please run: pip install mblt_model_zoo[transformers]"
    )

__all__ = ["MobilintLlamaConfig", "MobilintLlamaForCausalLM"]