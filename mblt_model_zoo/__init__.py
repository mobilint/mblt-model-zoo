"""Deprecated Mobilint Model Zoo compatibility package.

``mblt-model-zoo`` no longer receives updates. Use the standalone packages instead: ``mblt-vision-python``
(``mblt_vision``), ``transformers-mblt`` (``transformers_mblt``), ``melotts-mblt`` (``melotts_mblt``), and
``mblt-npu-python`` (``mblt_npu``). The subpackages below are imported lazily so that each one emits its own
deprecation warning naming its replacement only when it is used.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

__version__ = "2.13.0"

_SUBPACKAGES = frozenset({"utils", "vision", "compile", "hf_transformers", "MeloTTS"})


def _is_installed(module_name: str) -> bool:
    """Return whether an optional standalone package can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


__all__ = ["utils", "vision"]
if _is_installed("transformers_mblt"):
    __all__.append("hf_transformers")
if _is_installed("melotts_mblt"):
    __all__.append("MeloTTS")


def __getattr__(name: str) -> Any:
    """Import compatibility subpackages on first attribute access."""
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBPACKAGES)
