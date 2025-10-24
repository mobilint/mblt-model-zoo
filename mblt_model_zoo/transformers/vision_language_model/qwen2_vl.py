from typing import Optional, Union, Any, TypeVar

import maccel
import os
import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
from cv2 import resize as cv2_resize, INTER_CUBIC
import torch.nn.functional as F

from transformers import (
    Qwen2VLTextConfig,
    Qwen2VLConfig,
    Qwen2TokenizerFast,
    Qwen2VLProcessor,
    Qwen2VLPreTrainedModel,
    Qwen2VLModel,
    Qwen2VLForConditionalGeneration,

    PretrainedConfig,
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
)
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.image_utils import ImageInput
from transformers.video_utils import VideoInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessorKwargs
from transformers.image_utils import load_image

from mblt_model_zoo.transformers.utils.generation_utils import MobilintGenerationMixin

from ..utils.cache_utils import MobilintCache


logger = logging.get_logger(__name__)


# type hinting: specifying the type of config class that inherits from PretrainedConfig
SpecificPretrainedConfigType = TypeVar("SpecificPretrainedConfigType", bound="PretrainedConfig")


class MobilintQwen2VLProcessor(Qwen2VLProcessor):
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        # Make sure images is only one instance of PIL.Image.Image, np.ndarray, torch.Tensor, or None
        while isinstance(images, list):
            if len(images) > 1:
                raise NotImplementedError("Only one image input is supported")
            images = images[0]
        
        if isinstance(images, str):
            images = load_image(images)
        
        # Image should be resized into (224, 224) to fit image token position
        size = (224, 224)
        
        if isinstance(images, Image.Image):
            images = images.resize(size)
        elif isinstance(images, np.ndarray):
            if images.ndim == 2:  # 흑백
                return cv2_resize(images, size[::-1], interpolation=INTER_CUBIC)
            elif images.ndim == 3:
                return cv2_resize(images, size[::-1], interpolation=INTER_CUBIC)
            else:
                raise ValueError(f"Unsupported ndarray shape: {images.shape}")
        elif torch.is_tensor(images):
            if images.ndim == 3:  # CHW
                images = images.unsqueeze(0).float()  # BCHW
                images = F.interpolate(images, size=size, mode="bicubic", align_corners=False)
            elif images.ndim == 2:  # HW
                images = images.unsqueeze(0).unsqueeze(0).float()  # B1HW
                images = F.interpolate(images, size=size, mode="bicubic", align_corners=False)
            else:
                raise ValueError(f"Unsupported tensor shape: {tuple(images.shape)}")
        else:
            raise TypeError(f"Unsupported type of image: {type(images)}")
        
        if videos is not None:
            raise NotImplementedError("Video inputs are not supported")
                    
        return super().__call__(images, text, videos, **kwargs)

class MobilintQwen2VLTextConfig(Qwen2VLTextConfig):
    model_type = "mobilint-qwen2_vl_text"
    keys_to_ignore_at_inference = []
    base_model_tp_plan = {}
    base_model_pp_plan = {}

    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        **kwargs,
    ):
        self.mxq_path = mxq_path
        self.dev_no = dev_no

        super().__init__(**kwargs)

        self.tie_word_embeddings = False


class MobilintQwen2VLVisionConfig(Qwen2VLVisionConfig):
    model_type = "mobilint-qwen2_vl"
    
    def __init__(
        self,
        mxq_path: str = "",
        dev_no: int = 0,
        **kwargs,
    ):
        self.mxq_path = mxq_path
        self.dev_no = dev_no

        super().__init__(**kwargs)
        

class MobilintQwen2VLConfig(Qwen2VLConfig):
    model_type = "mobilint-qwen2_vl"
    sub_configs = {"vision_config": MobilintQwen2VLVisionConfig, "text_config": MobilintQwen2VLTextConfig}
    keys_to_ignore_at_inference = []
    
    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> SpecificPretrainedConfigType:
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        
        config, kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs)
        
        config.text_config.name_or_path = config.name_or_path
        config.vision_config.name_or_path = config.name_or_path
        
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config
        
        
class MobilintQwen2VLPreTrainedModel(Qwen2VLPreTrainedModel):
    config: MobilintQwen2VLConfig
    supports_gradient_checkpointing = False
    _no_split_modules = []
    _supports_flash_attn = False
    _supports_sdpa = False

    _can_compile_fullgraph = False
    _supports_attention_backend = False


class MobilintQwen2VisionTransformerPretrainedModel(MobilintQwen2VLPreTrainedModel):
    config: MobilintQwen2VLVisionConfig
    _no_split_modules = []
    
    def __init__(self, config: MobilintQwen2VLVisionConfig) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.gradient_checkpointing = False
                
        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_multi_core_mode([maccel.Cluster.Cluster1])
        self.mxq_model = maccel.Model(f"{config.name_or_path}/{config.mxq_path}", mc)
        self.mxq_model.launch(self.acc)
    
    def __getattribute__(self, name: str, /) -> Any:
        if name == 'dtype':
            return self.get_dtype()
        elif name == 'device':
            return self.get_device()
        else:
            return super().__getattribute__(name)
    
    def get_dtype(self) -> torch.dtype:
        return torch.float32

    def get_device(self) -> torch.device:
        return 'cpu'

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        gt, gh, gw = grid_thw[0].tolist()

        c  = 3
        pt = 2
        Mh = 2
        Mw = 2
        gh2 = gh // 2
        gw2 = gw // 2
        ph = pw = int((hidden_states.shape[-1] // (pt * c)) ** 0.5)

        assert hidden_states.shape[0] == gt * gh2 * gw2 * Mh * Mw
        assert hidden_states.shape[1] == c * pt * ph * pw

        # (gt, gh2, gw2, Mh, Mw, c, pt, ph, pw)
        hidden_states = hidden_states.view(gt, gh2, gw2, Mh, Mw, c, pt, ph, pw)

        # rearrange: "(gt gh gw Mh Mw) (c pt ph pw) -> gt (gh gw ph) (Mh Mw pw) (pt c)"
        # (gt, pt, c, Mh, Mw, pw, gh2, gw2, ph)
        hidden_states = hidden_states.permute(0, 1, 2, 7, 3, 4, 8, 6, 5).contiguous()

        # gt (gh gw ph) (Mh Mw pw) (pt c)
        hidden_states = hidden_states.view(
            gt,
            gh2 * gw2 * ph,
            Mh * Mw * pw,
            pt * c,
        ).squeeze(0)

        hidden_states = hidden_states.to(torch.float32).cpu().numpy()
        image_embeds = self.mxq_model.infer(hidden_states)[0]
        image_embeds = torch.tensor(image_embeds, dtype=torch.float32, device=self.device).squeeze(0).squeeze(0)
        
        return image_embeds

    def dispose(self):
        self.mxq_model.dispose()


class MobilintQwen2VLTextModel(MobilintQwen2VLPreTrainedModel):
    config: MobilintQwen2VLTextConfig
    
    def __init__(self, config: MobilintQwen2VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.gradient_checkpointing = False
        
        self.dev_no = config.dev_no
        self.acc = maccel.Accelerator(self.dev_no)
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(1)
        self.mxq_model = maccel.Model(f"{config.name_or_path}/{config.mxq_path}", mc)
        self.mxq_model.launch(self.acc)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        chunk_size: int = 0,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if output_attentions:
            logger.warning_once("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning_once("output_hidden_states is not supported.")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        inputs_embeds = inputs_embeds.type(torch.float32).cpu().numpy()

        if inputs_embeds.ndim == 3:
            inputs_embeds = np.expand_dims(
                inputs_embeds, 1
            )  # (batch, 1, seqlen, hidden_size)

        # max width should be appropriate number for chunking (ex. 192 for Llama 3.2 3B)
        # it should be searched experimentally
        if chunk_size == 0:
            chunk_size = self.mxq_model.get_input_buffer_info()[0].max_width
        num_of_chunks = math.ceil(inputs_embeds.shape[2] / chunk_size)

        for i in range(num_of_chunks):
            start_index = i * chunk_size
            end_index = min(start_index + chunk_size, inputs_embeds.shape[2])
            cache_size = 0 if past_key_values is None else past_key_values.get_seq_length()

            logits = self.mxq_model.infer([inputs_embeds[:, :, start_index:end_index, :]], None, cache_size)[0]

            if use_cache:
                past_key_values.update_cache_position(cache_position[start_index:end_index])

        logits = torch.tensor(logits, dtype=torch.float32, device=self.device).squeeze(0)

        if not return_dict:
            return tuple(
                v for v in [logits, past_key_values] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
    
    def dispose(self):
        self.mxq_model.dispose()


class MobilintQwen2VLModel(MobilintQwen2VLPreTrainedModel, Qwen2VLModel):
    def __init__(self, config: MobilintQwen2VLConfig):
        MobilintQwen2VLPreTrainedModel.__init__(self, config)
        self.visual = MobilintQwen2VisionTransformerPretrainedModel(config.vision_config)
        self.language_model = MobilintQwen2VLTextModel(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here
    
    def dispose(self):
        self.visual.dispose()
        self.language_model.dispose()


class MobilintQwen2VLForConditionalGeneration(MobilintQwen2VLPreTrainedModel, MobilintGenerationMixin, Qwen2VLForConditionalGeneration):
    _tied_weights_keys = []
    
    def __init__(self, config: MobilintQwen2VLConfig):
        MobilintQwen2VLPreTrainedModel.__init__(self, config)
        
        self.model = MobilintQwen2VLModel(config)
        # lm_head is done in self.model
        # So we just replace self.lm_head with identity module
        self.lm_head = nn.Identity()
    
    def get_mxq_model(self):
        return self.model.language_model.mxq_model
    
    def dispose(self):
        self.model.dispose()
        

AutoConfig.register("mobilint-qwen2_vl", MobilintQwen2VLConfig)
AutoTokenizer.register(MobilintQwen2VLConfig, fast_tokenizer_class=Qwen2TokenizerFast)
AutoProcessor.register(MobilintQwen2VLConfig, MobilintQwen2VLProcessor)
AutoModelForImageTextToText.register(MobilintQwen2VLConfig, MobilintQwen2VLForConditionalGeneration)

from ..utils.types import TransformersModelInfo

qwen2_vl_2b_instruct = TransformersModelInfo(
    original_model_id="Qwen/Qwen2-VL-2B-Instruct",
    model_id="mobilint/Qwen2-VL-2B-Instruct",
    download_url_base="https://dl.mobilint.com/model/transformers/vlm/Qwen2-VL-2B-Instruct/",
    file_list=[
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "preprocessor_config.json",
        "Qwen2-VL-2B-Instruct_text_model.mxq",
        "Qwen2-VL-2B-Instruct_vision_transformer.mxq",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ],
)
