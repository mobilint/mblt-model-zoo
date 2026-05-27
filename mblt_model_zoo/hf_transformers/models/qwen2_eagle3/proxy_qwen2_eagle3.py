try:
    from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.configuration_qwen2_eagle3 import (
        MobilintQwen2Eagle3Config,
    )
    from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
        MobilintQwen2Eagle3ForCausalLM,
    )
except ImportError as exc:
    raise ImportError(
        "Failed to import Mobilint Qwen2 EAGLE-3 modules. "
        "Install mblt-model-zoo with the transformers extra."
    ) from exc


__all__ = ["MobilintQwen2Eagle3Config", "MobilintQwen2Eagle3ForCausalLM"]
