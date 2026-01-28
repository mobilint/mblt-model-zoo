from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.whisper.configuration_whisper import WhisperConfig

from ...utils.configuration_utils import MobilintEncoderDecoderConfigMixin


class MobilintWhisperConfig(MobilintEncoderDecoderConfigMixin, WhisperConfig):
    model_type = "mobilint-whisper"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-whisper", MobilintWhisperConfig)
