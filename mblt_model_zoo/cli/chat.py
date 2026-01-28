from argparse import Namespace

def register_mobilint_models(args: Namespace, transformers):
    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path_or_address,
        trust_remote_code=True,
    )

    model_type = getattr(config, "model_type", "")
    arch_name = config.architectures[0] if getattr(config, "architectures", None) else ""
    
    if model_type.startswith("mobilint-") or arch_name.startswith("Mobilint"):
        original_model_type = model_type[len("mobilint-"):] if model_type.startswith("mobilint-") else model_type

        import importlib
        module = importlib.import_module(
            f"mblt_model_zoo.hf_transformers.models.{original_model_type}.modeling_{original_model_type}"
        )
        setattr(transformers, config.architectures[0], module.__dict__[config.architectures[0]])
        
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = transformers.models.auto.modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

        if arch_name.endswith("CausalLM"):
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type] = arch_name
        elif arch_name.endswith("ConditionalGeneration"):
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES[model_type] = arch_name