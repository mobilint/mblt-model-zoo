"""Shared helpers for the Model Zoo CLI bridges to the standalone ``transformers-mblt`` package."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Callable, NoReturn

INSTALL_HINT = (
    "Transformers commands are provided by the standalone transformers-mblt package. "
    "Install it with: pip install transformers-mblt (or pip install 'mblt-model-zoo[transformers]')"
)


def load_standalone(module_name: str) -> ModuleType | None:
    """Import ``transformers_mblt.<module_name>``, returning ``None`` only when transformers-mblt is not installed."""
    try:
        return importlib.import_module(f"transformers_mblt.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name != "transformers_mblt":
            raise
        return None


def exit_missing_dependency() -> int:
    """Report the missing transformers-mblt dependency and return the CLI usage-error status."""
    print(INSTALL_HINT, file=sys.stderr)
    return 2


def missing_dependency_getattr(module_name: str) -> Callable[[str], NoReturn]:
    """Build a module ``__getattr__`` that reports the missing transformers-mblt dependency on first use.

    The bridge module itself stays importable; using any of its names (``from ... import name``) raises
    ``ModuleNotFoundError`` with :data:`INSTALL_HINT`. Dunder lookups raise ``AttributeError`` so ``hasattr`` and
    introspection keep working.
    """

    def __getattr__(name: str) -> NoReturn:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        raise ModuleNotFoundError(f"{module_name}.{name}: {INSTALL_HINT}", name="transformers_mblt")

    return __getattr__
