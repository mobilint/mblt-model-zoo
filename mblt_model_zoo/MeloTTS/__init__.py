"""Deprecated compatibility facade for :mod:`melotts_mblt`.

MeloTTS is maintained in ``melotts-mblt``. Every ``mblt_model_zoo.MeloTTS.<path>`` import resolves to the *same module
object* as ``melotts_mblt.<path>``, so existing code such as ``from mblt_model_zoo.MeloTTS.api import TTS`` keeps
working. New code should use ``from melotts_mblt import TTS``.
"""

from __future__ import annotations

from typing import Any

from .._deprecation import warn_deprecated_import
from .._standalone_alias import install_alias

try:
    import melotts_mblt as _standalone
except ModuleNotFoundError as exc:
    if exc.name != "melotts_mblt":
        raise
    raise ModuleNotFoundError(
        "mblt_model_zoo.MeloTTS now forwards to the standalone melotts-mblt package. "
        "Install it with: pip install melotts-mblt (or pip install 'mblt-model-zoo[MeloTTS]')",
        name=exc.name,
    ) from exc

warn_deprecated_import(__name__, "melotts-mblt", "melotts_mblt")
install_alias(__name__, _standalone.__name__)

__all__ = list(_standalone.__all__)


def __getattr__(name: str) -> Any:
    """Forward package attributes (``TTS`` ...) to :mod:`melotts_mblt`."""
    return getattr(_standalone, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
