import os
from typing import Any, Literal, Optional, TypeVar, Union, overload

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
from transformers.pipelines.audio_classification import AudioClassificationPipeline
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)
from transformers.pipelines.base import Pipeline
from transformers.pipelines.depth_estimation import DepthEstimationPipeline
from transformers.pipelines.document_question_answering import (
    DocumentQuestionAnsweringPipeline,
)
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from transformers.pipelines.fill_mask import FillMaskPipeline
from transformers.pipelines.image_classification import ImageClassificationPipeline
from transformers.pipelines.image_feature_extraction import (
    ImageFeatureExtractionPipeline,
)
from transformers.pipelines.image_segmentation import ImageSegmentationPipeline
from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline
from transformers.pipelines.image_to_image import ImageToImagePipeline
from transformers.pipelines.image_to_text import ImageToTextPipeline
from transformers.pipelines.keypoint_matching import KeypointMatchingPipeline
from transformers.pipelines.mask_generation import MaskGenerationPipeline
from transformers.pipelines.object_detection import ObjectDetectionPipeline
from transformers.pipelines.question_answering import QuestionAnsweringPipeline
from transformers.pipelines.table_question_answering import (
    TableQuestionAnsweringPipeline,
)
from transformers.pipelines.text2text_generation import (
    SummarizationPipeline,
    Text2TextGenerationPipeline,
    TranslationPipeline,
)
from transformers.pipelines.text_classification import TextClassificationPipeline
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.pipelines.text_to_audio import TextToAudioPipeline
from transformers.pipelines.token_classification import TokenClassificationPipeline
from transformers.pipelines.video_classification import VideoClassificationPipeline
from transformers.pipelines.visual_question_answering import (
    VisualQuestionAnsweringPipeline,
)
from transformers.pipelines.zero_shot_audio_classification import (
    ZeroShotAudioClassificationPipeline,
)
from transformers.pipelines.zero_shot_classification import (
    ZeroShotClassificationPipeline,
)
from transformers.pipelines.zero_shot_image_classification import (
    ZeroShotImageClassificationPipeline,
)
from transformers.pipelines.zero_shot_object_detection import (
    ZeroShotObjectDetectionPipeline,
)

from ...utils import download_url_to_folder
from ._api import list_models

T = TypeVar('T')

def convert_identifier_to_path(identifier: Union[str, T]) -> Union[str, T]:
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


@overload
def pipeline(task: Literal[None], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> Pipeline: ...
@overload
def pipeline(task: Literal["audio-classification"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> AudioClassificationPipeline: ...
@overload
def pipeline(task: Literal["automatic-speech-recognition"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> AutomaticSpeechRecognitionPipeline: ...
@overload
def pipeline(task: Literal["depth-estimation"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> DepthEstimationPipeline: ...
@overload
def pipeline(task: Literal["document-question-answering"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> DocumentQuestionAnsweringPipeline: ...
@overload
def pipeline(task: Literal["feature-extraction"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> FeatureExtractionPipeline: ...
@overload
def pipeline(task: Literal["fill-mask"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> FillMaskPipeline: ...
@overload
def pipeline(task: Literal["image-classification"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ImageClassificationPipeline: ...
@overload
def pipeline(task: Literal["image-feature-extraction"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ImageFeatureExtractionPipeline: ...
@overload
def pipeline(task: Literal["image-segmentation"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ImageSegmentationPipeline: ...
@overload
def pipeline(task: Literal["image-text-to-text"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ImageTextToTextPipeline: ...
@overload
def pipeline(task: Literal["image-to-image"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ImageToImagePipeline: ...
@overload
def pipeline(task: Literal["image-to-text"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ImageToTextPipeline: ...
@overload
def pipeline(task: Literal["keypoint-matching"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> KeypointMatchingPipeline: ...
@overload
def pipeline(task: Literal["mask-generation"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> MaskGenerationPipeline: ...
@overload
def pipeline(task: Literal["object-detection"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ObjectDetectionPipeline: ...
@overload
def pipeline(task: Literal["question-answering"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> QuestionAnsweringPipeline: ...
@overload
def pipeline(task: Literal["summarization"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> SummarizationPipeline: ...
@overload
def pipeline(task: Literal["table-question-answering"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> TableQuestionAnsweringPipeline: ...
@overload
def pipeline(task: Literal["text-classification"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> TextClassificationPipeline: ...
@overload
def pipeline(task: Literal["text-generation"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> TextGenerationPipeline: ...
@overload
def pipeline(task: Literal["text-to-audio"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> TextToAudioPipeline: ...
@overload
def pipeline(task: Literal["text2text-generation"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> Text2TextGenerationPipeline: ...
@overload
def pipeline(task: Literal["token-classification"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> TokenClassificationPipeline: ...
@overload
def pipeline(task: Literal["translation"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> TranslationPipeline: ...
@overload
def pipeline(task: Literal["video-classification"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> VideoClassificationPipeline: ...
@overload
def pipeline(task: Literal["visual-question-answering"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> VisualQuestionAnsweringPipeline: ...
@overload
def pipeline(task: Literal["zero-shot-audio-classification"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ZeroShotAudioClassificationPipeline: ...
@overload
def pipeline(task: Literal["zero-shot-classification"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ZeroShotClassificationPipeline: ...
@overload
def pipeline(task: Literal["zero-shot-image-classification"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ZeroShotImageClassificationPipeline: ...
@overload
def pipeline(task: Literal["zero-shot-object-detection"], model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None, config: Optional[Union[str, PretrainedConfig]] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None, feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None, image_processor: Optional[Union[str, BaseImageProcessor]] = None, processor: Optional[Union[str, ProcessorMixin]] = None, framework: Optional[str] = None, revision: Optional[str] = None, use_fast: bool = True, token: Optional[Union[str, bool]] = None, device: Optional[Union[int, str, "torch.device"]] = None, device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, dtype: Optional[Union[str, "torch.dtype"]] = "auto", trust_remote_code: Optional[bool] = None, model_kwargs: Optional[dict[str, Any]] = None, pipeline_class: Optional[Any] = None, **kwargs: Any) -> ZeroShotObjectDetectionPipeline: ...


def pipeline(
    task: Optional[str] = None,
    model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    image_processor: Optional[Union[str, BaseImageProcessor]] = None,
    processor: Optional[Union[str, ProcessorMixin]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None,
    dtype: Optional[Union[str, "torch.dtype"]] = "auto",
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs: Any,
) -> Pipeline:
    model = convert_identifier_to_path(model)
    config = convert_identifier_to_path(config)
    tokenizer = convert_identifier_to_path(tokenizer)
    feature_extractor = convert_identifier_to_path(feature_extractor)
    image_processor = convert_identifier_to_path(image_processor)
    processor = convert_identifier_to_path(processor)

    return original_pipeline(
        task, # type: ignore
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
        dtype,
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
