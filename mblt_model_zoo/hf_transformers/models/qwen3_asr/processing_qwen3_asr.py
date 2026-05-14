"""Re-export of upstream ``Qwen3ASRProcessor`` under a Mobilint-friendly name."""

from ._errors import guard_qwen_asr_import

with guard_qwen_asr_import():
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

MobilintQwen3ASRProcessor = Qwen3ASRProcessor

__all__ = ["MobilintQwen3ASRProcessor"]
