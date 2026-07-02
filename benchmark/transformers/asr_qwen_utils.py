"""Qwen3-ASR-specific helpers shared by ASR benchmark scripts.

These utilities isolate optional upstream ``qwen_asr`` integration details so the
main benchmark CLI stays focused on orchestration.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Mapping

import torch

from mblt_model_zoo.hf_transformers.models.qwen3_asr._errors import QWEN_ASR_INSTALL_HINT


_QWEN3_ASR_LANGUAGE_ALIASES = {
    "ar": "Arabic",
    "arabic": "Arabic",
    "cantones": "Cantonese",
    "cantonese": "Cantonese",
    "chinese": "Chinese",
    "cs": "Czech",
    "cz": "Czech",
    "czech": "Czech",
    "da": "Danish",
    "danish": "Danish",
    "de": "German",
    "deutsch": "German",
    "dutch": "Dutch",
    "el": "Greek",
    "en": "English",
    "eng": "English",
    "english": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "filipino": "Filipino",
    "finnish": "Finnish",
    "fr": "French",
    "french": "French",
    "german": "German",
    "greek": "Greek",
    "hi": "Hindi",
    "hindi": "Hindi",
    "hu": "Hungarian",
    "hungarian": "Hungarian",
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "it": "Italian",
    "italian": "Italian",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "kor": "Korean",
    "korean": "Korean",
    "macedonian": "Macedonian",
    "malay": "Malay",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "persian": "Persian",
    "pl": "Polish",
    "polish": "Polish",
    "portuguese": "Portuguese",
    "pt": "Portuguese",
    "ro": "Romanian",
    "romanian": "Romanian",
    "ru": "Russian",
    "russian": "Russian",
    "spanish": "Spanish",
    "sv": "Swedish",
    "swedish": "Swedish",
    "th": "Thai",
    "thai": "Thai",
    "tr": "Turkish",
    "turkish": "Turkish",
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
}


def is_qwen3_asr_model(model_id: str) -> bool:
    """Return whether the model id refers to a Qwen3-ASR checkpoint."""

    normalized = model_id.lower()
    return "qwen3-asr" in normalized or "qwen3_asr" in normalized


def resolve_native_qwen3_asr_language(language: str | None) -> str | None:
    """Return the native Qwen3-ASR language name for a CLI language hint."""

    if language is None:
        return None
    text = str(language).strip()
    if not text:
        return None
    normalized = text.casefold().replace("_", "-")
    return _QWEN3_ASR_LANGUAGE_ALIASES.get(normalized, text)


def ensure_qwen3_asr_backend_registered() -> None:
    """Register upstream Qwen3-ASR Transformers backend metadata when available.

    Raises:
        ModuleNotFoundError: If the optional ``qwen_asr`` package is required but missing.
    """

    try:
        import qwen_asr.core.transformers_backend  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "qwen_asr" or missing.startswith("qwen_asr."):
            raise ModuleNotFoundError(QWEN_ASR_INSTALL_HINT) from exc
        raise

    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
        from transformers.models.auto.modeling_auto import AutoModelForSpeechSeq2Seq

        Qwen3ASRForConditionalGeneration.main_input_name = "input_features"
        AutoModelForSpeechSeq2Seq.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)

        try:
            from transformers.pipelines.automatic_speech_recognition import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
        except (ImportError, AttributeError):
            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = None

        if isinstance(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES, dict):
            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.setdefault(
                "qwen3_asr",
                "Qwen3ASRForConditionalGeneration",
            )
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "qwen_asr" or missing.startswith("qwen_asr."):
            raise ModuleNotFoundError(QWEN_ASR_INSTALL_HINT) from exc
        raise


def configure_native_qwen3_asr_generate(pipe: Any, generate_kwargs: Mapping[str, Any] | None) -> Any:
    """Attach benchmark generation overrides to native Qwen3-ASR objects."""

    if not generate_kwargs:
        ensure_native_qwen3_asr_generation_config(pipe)
        return pipe

    ensure_native_qwen3_asr_generation_config(pipe)
    inner_model = getattr(pipe, "model", None)
    original_generate = getattr(inner_model, "generate", None)
    if inner_model is None or not callable(original_generate):
        return pipe

    resolved_pad_token_id = resolve_native_qwen3_asr_pad_token_id(pipe)
    overrides = {
        key: value
        for key, value in dict(generate_kwargs).items()
        if key not in {"return_timestamps"} and value is not None
    }
    if resolved_pad_token_id is not None:
        overrides.setdefault("pad_token_id", resolved_pad_token_id)
    if not overrides:
        return pipe

    @functools.wraps(original_generate)
    def _generate_with_overrides(*args: Any, **kwargs: Any) -> Any:
        merged_kwargs = dict(overrides)
        merged_kwargs.update(kwargs)
        return original_generate(*args, **merged_kwargs)

    inner_model.generate = _generate_with_overrides
    return pipe


def ensure_native_qwen3_asr_generation_config(pipe: Any) -> Any:
    """Populate native Qwen3-ASR generation config defaults needed by benchmarks."""

    inner_model = getattr(pipe, "model", None)
    if inner_model is None:
        return pipe

    model_config = getattr(inner_model, "config", None)
    generation_config = getattr(inner_model, "generation_config", None)
    if generation_config is None:
        return pipe

    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None and model_config is not None:
        eos_token_id = getattr(model_config, "eos_token_id", None)

    pad_token_id = getattr(generation_config, "pad_token_id", None)
    if pad_token_id is None and model_config is not None:
        pad_token_id = getattr(model_config, "pad_token_id", None)

    if pad_token_id is None and eos_token_id is not None:
        generation_config.pad_token_id = eos_token_id
        if model_config is not None and getattr(model_config, "pad_token_id", None) is None:
            model_config.pad_token_id = eos_token_id

    resolved_pad_token_id = resolve_native_qwen3_asr_pad_token_id(pipe)
    if resolved_pad_token_id is not None:
        generation_config.pad_token_id = resolved_pad_token_id
        if model_config is not None:
            model_config.pad_token_id = resolved_pad_token_id

    return pipe


def resolve_native_qwen3_asr_pad_token_id(pipe: Any) -> int | list[int] | None:
    """Resolve a stable pad token id for native upstream Qwen3-ASR generation."""

    inner_model = getattr(pipe, "model", None)
    generation_config = getattr(inner_model, "generation_config", None) if inner_model is not None else None
    model_config = getattr(inner_model, "config", None) if inner_model is not None else None
    tokenizer = getattr(pipe, "tokenizer", None)

    candidates = [
        getattr(generation_config, "pad_token_id", None),
        getattr(model_config, "pad_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(generation_config, "eos_token_id", None),
        getattr(model_config, "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def resolve_torch_dtype(dtype: str | None) -> torch.dtype | None:
    """Resolve CLI dtype text into a torch dtype object when possible."""

    if dtype is None:
        return None
    text = str(dtype).strip()
    if not text:
        return None
    normalized = text.removeprefix("torch.")
    resolved = getattr(torch, normalized, None)
    return resolved if isinstance(resolved, torch.dtype) else None


def move_native_qwen3_asr_to_device(pipe: Any, *, device: str | None, device_map: str | None) -> Any:
    """Ensure native upstream Qwen3-ASR model is placed on the requested device."""

    if device_map:
        return pipe
    if not device:
        return pipe
    inner_model = getattr(pipe, "model", None)
    move_to = getattr(inner_model, "to", None)
    if not callable(move_to):
        return pipe
    move_to(device)
    return pipe


def quiet_apscheduler_info_logs() -> None:
    """Raise APScheduler logger level to avoid info-log noise from qwen_asr."""

    aps_logger = logging.getLogger("apscheduler")
    if aps_logger.level == logging.NOTSET or aps_logger.level < logging.WARNING:
        aps_logger.setLevel(logging.WARNING)


def supports_native_transcribe_language(pipe: Any) -> bool:
    """Return whether a native ASR transcribe callable accepts ``language``."""

    transcribe = getattr(pipe, "transcribe", None)
    if not callable(transcribe):
        return False
    try:
        signature = inspect.signature(transcribe)
    except (TypeError, ValueError):
        return False
    return "language" in signature.parameters