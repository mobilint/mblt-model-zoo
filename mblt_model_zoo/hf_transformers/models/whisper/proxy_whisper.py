try:
    from mblt_model_zoo.hf_transformers.models.whisper.configuration_whisper import (
        MobilintWhisperConfig,
    )
    from mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper import (
        MobilintWhisperForConditionalGeneration,
    )
except ImportError:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. "
        "Please run: pip install mblt_model_zoo[transformers]"
    )

__all__ = ["MobilintWhisperConfig", "MobilintWhisperForConditionalGeneration"]