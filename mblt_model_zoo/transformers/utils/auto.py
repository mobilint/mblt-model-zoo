import os
from typing import Any, Dict, Optional, Union

import torch
from transformers import AutoConfig as OriginalAutoConfig
from transformers import AutoFeatureExtractor as OriginalAutoFeatureExtractor
from transformers import AutoModel as OriginalAutoModel
from transformers import AutoModelForCausalLM as OriginalAutoModelForCausalLM
from transformers import (
    AutoModelForImageTextToText as OriginalAutoModelForImageTextToText,
)
from transformers import AutoModelForMaskedLM as OriginalAutoModelForMaskedLM
from transformers import AutoModelForSpeechSeq2Seq as OriginalAutoModelForSpeechSeq2Seq
from transformers import AutoModelForVision2Seq as OriginalAutoModelForVision2Seq
from transformers import AutoProcessor as OriginalAutoProcessor
from transformers import AutoTokenizer as OriginalAutoTokenizer
from transformers import (
    BaseImageProcessor,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    ProcessorMixin,
    TFPreTrainedModel,
)
from transformers import pipeline as original_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.pipelines.base import Pipeline

from ...utils import download_url_to_folder
from .._api import list_models


def convert_identifier_to_path(identifier: Any):
    if not isinstance(identifier, str):
        return identifier

    filtered = [
        model
        for models in list_models().values()
        for model in models
        if model.model_id == identifier
    ]
    model = filtered[0] if filtered else None

    if model is None:
        return identifier

    download_path = os.path.expanduser(
        f"~/.mblt_model_zoo/transformers/{model.get_directory_name()}/"
    )
    download_url_to_folder(model.download_url_base, model.file_list, download_path)

    return download_path


def pipeline(
    task: Optional[str] = None,
    model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[
        Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]
    ] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    image_processor: Optional[Union[str, BaseImageProcessor]] = None,
    processor: Optional[Union[str, ProcessorMixin]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map=None,
    torch_dtype=None,
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs,
) -> Pipeline:
    model = convert_identifier_to_path(model)
    config = convert_identifier_to_path(config)
    tokenizer = convert_identifier_to_path(tokenizer)
    feature_extractor = convert_identifier_to_path(feature_extractor)
    image_processor = convert_identifier_to_path(image_processor)
    processor = convert_identifier_to_path(processor)

    return original_pipeline(
        task,
        model,
        config,
        tokenizer,
        feature_extractor,
        image_processor,
        processor,
        framework,
        revision,
        use_fast,
        token,
        device,
        device_map,
        torch_dtype,
        trust_remote_code,
        model_kwargs,
        pipeline_class,
        **kwargs,
    )


class AutoConfig(OriginalAutoConfig):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        pretrained_model_name_or_path = convert_identifier_to_path(
            pretrained_model_name_or_path
        )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


class AutoModel(OriginalAutoModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        pretrained_model_name_or_path = convert_identifier_to_path(
            pretrained_model_name_or_path
        )
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


class AutoTokenizer(OriginalAutoTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        pretrained_model_name_or_path = convert_identifier_to_path(
            pretrained_model_name_or_path
        )
        return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)


class AutoFeatureExtractor(OriginalAutoFeatureExtractor):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        pretrained_model_name_or_path = convert_identifier_to_path(
            pretrained_model_name_or_path
        )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


class AutoProcessor(OriginalAutoProcessor):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        pretrained_model_name_or_path = convert_identifier_to_path(
            pretrained_model_name_or_path
        )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


class AutoModelForSpeechSeq2Seq(OriginalAutoModelForSpeechSeq2Seq):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        pretrained_model_name_or_path = convert_identifier_to_path(
            pretrained_model_name_or_path
        )
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


class AutoModelForImageTextToText(OriginalAutoModelForImageTextToText):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        pretrained_model_name_or_path = convert_identifier_to_path(
            pretrained_model_name_or_path
        )
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


class AutoModelForMaskedLM(OriginalAutoModelForMaskedLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        pretrained_model_name_or_path = convert_identifier_to_path(
            pretrained_model_name_or_path
        )
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


class AutoModelForCausalLM(OriginalAutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        pretrained_model_name_or_path = convert_identifier_to_path(
            pretrained_model_name_or_path
        )
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


class AutoModelForVision2Seq(OriginalAutoModelForVision2Seq):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        pretrained_model_name_or_path = convert_identifier_to_path(
            pretrained_model_name_or_path
        )
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
