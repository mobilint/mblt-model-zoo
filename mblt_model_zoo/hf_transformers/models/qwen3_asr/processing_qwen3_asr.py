"""Re-export of upstream ``Qwen3ASRProcessor`` under a Mobilint-friendly name."""

from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

MobilintQwen3ASRProcessor = Qwen3ASRProcessor

__all__ = ["MobilintQwen3ASRProcessor"]
