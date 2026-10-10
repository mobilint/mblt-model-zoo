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
# Optional subpackages and the standalone package each one forwards to (installed by an extra).
_OPTIONAL_SUBPACKAGES = {"hf_transformers": "transformers_mblt", "MeloTTS": "melotts_mblt"}


def _is_installed(module_name: str) -> bool:
    """Return whether an optional standalone package can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


__all__ = ["utils", "vision"]
__all__ += [name for name, package in _OPTIONAL_SUBPACKAGES.items() if _is_installed(package)]


def __getattr__(name: str) -> Any:
    """Import compatibility subpackages on first attribute access."""
    if name in _SUBPACKAGES:
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            # A missing optional extra means the attribute does not exist, so hasattr() and getattr(..., default)
            # keep working. The facade's install hint is preserved as the cause; import it directly to see it.
            if exc.name != _OPTIONAL_SUBPACKAGES.get(name):
                raise
            raise AttributeError(f"module {__name__!r} has no attribute {name!r} ({exc})") from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    available = _SUBPACKAGES - {name for name in _OPTIONAL_SUBPACKAGES if name not in __all__}
    return sorted(set(globals()) | available)
