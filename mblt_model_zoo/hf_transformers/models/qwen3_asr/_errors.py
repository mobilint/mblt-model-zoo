"""Helpers to convert ``qwen_asr`` ImportError into actionable install guidance."""

from contextlib import contextmanager
from typing import Iterator

QWEN_ASR_INSTALL_HINT = (
    "Mobilint Qwen3-ASR requires the upstream 'qwen-asr' package, "
    "which is exposed through the optional mblt-model-zoo 'qwen-asr' extra "
    "and is not bundled with mblt_model_zoo[transformers]. "
    "Install it with: uv sync --extra qwen-asr, or "
    "pip install -U \"mblt-model-zoo[qwen-asr]\" "
    "(see https://huggingface.co/Qwen/Qwen3-ASR-1.7B for details)."
)


@contextmanager
def guard_qwen_asr_import() -> Iterator[None]:
    """Re-raise ``ModuleNotFoundError`` for ``qwen_asr`` with an install hint.

    Other ``ModuleNotFoundError`` causes (e.g. a different missing dependency
    inside ``qwen_asr``'s transitive graph) are propagated unchanged so that
    the original traceback and missing-module name remain visible for
    debugging.
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "qwen_asr" or missing.startswith("qwen_asr."):
            raise ModuleNotFoundError(QWEN_ASR_INSTALL_HINT) from exc
        raise
