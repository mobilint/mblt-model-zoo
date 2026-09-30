"""Shared helpers for the Model Zoo CLI bridges to the standalone ``transformers-mblt`` package."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

INSTALL_HINT = (
    "Transformers commands are provided by the standalone transformers-mblt package. "
    "Install it with: pip install 'mblt-model-zoo[transformers]'"
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
