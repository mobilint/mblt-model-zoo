from typing import Any
import copy

from transformers.configuration_utils import (
    PretrainedConfig,
    SpecificPretrainedConfigType,
)

from .base_utils import MobilintNPUBackend


class MobilintConfigMixin(PretrainedConfig):
    def __init__(
        self,
        **kwargs
    ):
        self.npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="")
        super().__init__(**kwargs)
    
    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        if hasattr(self, "npu_backend"):
            _ = d.pop("npu_backend", None)

        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()
        output.update(self.npu_backend.to_dict(prefix=""))
        return output


class MobilintEncoderDecoderConfigMixin(PretrainedConfig):
    def __init__(
        self,
        **kwargs
    ):
        self.encoder_npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="encoder_")
        self.decoder_npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="decoder_")
        super().__init__(**kwargs)
        
    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        if hasattr(self, "encoder_npu_backend"):
            _ = d.pop("encoder_npu_backend", None)
        
        if hasattr(self, "decoder_npu_backend"):
            _ = d.pop("decoder_npu_backend", None)

        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()

        output.update(self.encoder_npu_backend.to_dict(prefix="encoder_"))
        output.update(self.decoder_npu_backend.to_dict(prefix="decoder_"))

        return output

    def get_text_config(self, decoder=None, encoder=None) -> "PretrainedConfig":
        return_both = decoder == encoder  # both unset or both set -> search all possible names

        decoder_possible_text_config_names = ("decoder", "generator", "text_config")
        encoder_possible_text_config_names = ("text_encoder",)
        if return_both:
            possible_text_config_names = encoder_possible_text_config_names + decoder_possible_text_config_names
        elif decoder:
            possible_text_config_names = decoder_possible_text_config_names
        else:
            possible_text_config_names = encoder_possible_text_config_names

        valid_text_config_names = []
        for text_config_name in possible_text_config_names:
            if hasattr(self, text_config_name):
                text_config = getattr(self, text_config_name, None)
                if text_config is not None:
                    valid_text_config_names += [text_config_name]

        if len(valid_text_config_names) > 1:
            raise ValueError(
                f"Multiple valid text configs were found in the model config: {valid_text_config_names}. In this "
                "case, using `get_text_config()` would be ambiguous. Please specify the desired text config directly, "
                "e.g. `text_config = config.sub_config_name`"
            )
        elif len(valid_text_config_names) == 1:
            config_to_return = getattr(self, valid_text_config_names[0])
        else:
            config_to_return = self

        # handle legacy models with flat config structure, when we only want one of the configs
        if not return_both and len(valid_text_config_names) == 0 and config_to_return.is_encoder_decoder:
            config_to_return = copy.deepcopy(config_to_return)
            prefix_to_discard = "encoder" if decoder else "decoder"
            prefix_to_keep = "decoder" if decoder else "encoder"
            for key in config_to_return.to_dict():
                # NOTE: We don't want to discard the key if it is mapped from a different attribute name at read time
                if key.startswith(prefix_to_discard) and key not in config_to_return.attribute_map.values():
                    try:
                        delattr(config_to_return, key)
                    except:
                        pass
                if key.startswith(prefix_to_keep):
                    # [encoder/decoder]_layers -> num_hidden_layers
                    if key == prefix_to_keep + "_layers":
                        new_key = "num_hidden_layers"
                    # [encoder/decoder]_attention_heads -> num_attention_heads
                    elif key == prefix_to_keep + "_attention_heads":
                        new_key = "num_attention_heads"
                    # e.g. encoder_hidden_act -> hidden_act
                    else:
                        new_key = key[len(prefix_to_keep) + 1 :]

                    # Does the class map the new key into a different attribute name at read time? if so, let's write
                    # into that attribute instead
                    if new_key in config_to_return.attribute_map:
                        new_key = config_to_return.attribute_map[new_key]
                    
                    try:
                        value = getattr(config_to_return, key)
                        delattr(config_to_return, key)
                        setattr(config_to_return, new_key, value)
                    except:
                        pass

        return config_to_return
