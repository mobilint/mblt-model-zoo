"""Shared helpers for the Model Zoo CLI bridges to the standalone ``melotts-mblt`` package."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

INSTALL_HINT = (
    "MeloTTS commands are provided by the standalone melotts-mblt package. "
    "Install it with: pip install melotts-mblt (or pip install 'mblt-model-zoo[MeloTTS]')"
)


def load_standalone(module_name: str) -> ModuleType | None:
    """Import ``melotts_mblt.<module_name>``, returning ``None`` only when melotts-mblt is not installed."""
    try:
        return importlib.import_module(f"melotts_mblt.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name != "melotts_mblt":
            raise
        return None


def exit_missing_dependency() -> int:
    """Report the missing melotts-mblt dependency and return the CLI usage-error status."""
    print(INSTALL_HINT, file=sys.stderr)
    return 2
