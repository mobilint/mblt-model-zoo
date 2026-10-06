"""Deprecated compatibility facade for :mod:`transformers_mblt`.

The Hugging Face Transformers integration is maintained in ``transformers-mblt``. Every
``mblt_model_zoo.hf_transformers.<path>`` import resolves to the *same module object* as
``transformers_mblt.<path>``, so existing code keeps working and the Hub ``proxy_*.py`` files that import
``mblt_model_zoo.hf_transformers.models.<arch>...`` load the standalone classes and their Auto registrations.
New code should import from :mod:`transformers_mblt` directly.
"""

from __future__ import annotations

import importlib
from typing import Any

from .._standalone_alias import install_alias

try:
    import transformers_mblt as _standalone
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
        raise
    raise ModuleNotFoundError(
        "mblt_model_zoo.hf_transformers now forwards to the standalone transformers-mblt package. "
        "Install it with: pip install 'mblt-model-zoo[transformers]'",
        name=exc.name,
    ) from exc

install_alias(__name__, _standalone.__name__)

__all__ = list(_standalone.__all__)


def __getattr__(name: str) -> Any:
    """Forward package attributes (``models``, ``utils``, ``register`` ...) to :mod:`transformers_mblt`."""
    if name in {"models", "utils"}:
        return importlib.import_module(f"{__name__}.{name}")
    return getattr(_standalone, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
