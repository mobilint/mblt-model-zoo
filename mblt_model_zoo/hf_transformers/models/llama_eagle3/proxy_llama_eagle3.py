try:
    from mblt_model_zoo.hf_transformers.models.llama_eagle3.configuration_llama_eagle3 import (
        MobilintLlamaEagle3Config,
    )
    from mblt_model_zoo.hf_transformers.models.llama_eagle3.modeling_llama_eagle3 import (
        MobilintLlamaEagle3ForCausalLM,
    )
except ImportError as e:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        "Please run: pip install mblt_model_zoo[transformers]"
    ) from e


__all__ = ["MobilintLlamaEagle3Config", "MobilintLlamaEagle3ForCausalLM"]
