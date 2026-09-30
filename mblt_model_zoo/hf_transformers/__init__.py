"""Deprecated compatibility facade for :mod:`transformers_mblt`.

The Hugging Face Transformers integration is maintained in ``transformers-mblt``. Every
``mblt_model_zoo.hf_transformers.<path>`` import resolves to the *same module object* as
``transformers_mblt.<path>``, so existing code keeps working and the Hub ``proxy_*.py`` files that import
``mblt_model_zoo.hf_transformers.models.<arch>...`` load the standalone classes and their Auto registrations.
New code should import from :mod:`transformers_mblt` directly.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType
from typing import Any

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

_ALIAS_PREFIX = __name__ + "."
_TARGET_PREFIX = _standalone.__name__ + "."


class _AliasLoader(importlib.abc.Loader):
    """Return the already-importable ``transformers_mblt`` module instead of executing a copy."""

    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return importlib.import_module(self._target)

    def exec_module(self, module: ModuleType) -> None:
        """The target module is fully initialized by its own import; nothing to execute."""


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Map ``mblt_model_zoo.hf_transformers.<path>`` imports onto ``transformers_mblt.<path>``."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_ALIAS_PREFIX):
            return None
        target_name = _TARGET_PREFIX + fullname[len(_ALIAS_PREFIX) :]
        if importlib.util.find_spec(target_name) is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(target_name))


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

__all__ = list(_standalone.__all__)


def __getattr__(name: str) -> Any:
    """Forward package attributes (``models``, ``utils``, ``register`` ...) to :mod:`transformers_mblt`."""
    if name in {"models", "utils"}:
        return importlib.import_module(f"{__name__}.{name}")
    return getattr(_standalone, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
