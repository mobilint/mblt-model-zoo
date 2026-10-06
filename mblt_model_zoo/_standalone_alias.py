"""Import aliasing shared by the Model Zoo compatibility facades.

``mblt_model_zoo.hf_transformers`` and ``mblt_model_zoo.MeloTTS`` forward to standalone packages
(``transformers_mblt`` and ``melotts_mblt``). :func:`install_alias` makes every ``<alias>.<path>`` import resolve to the
*same module object* as ``<target>.<path>``, so legacy imports keep working without copied implementations.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType
from typing import Any


class _AliasLoader(importlib.abc.Loader):
    """Return the already-importable target module instead of executing a copy."""

    def __init__(self, target: str) -> None:
        self._target = target
        self._target_spec: importlib.machinery.ModuleSpec | None = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self._target)
        self._target_spec = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        """Restore the target's own ``__spec__``, which the import system overwrote with the alias spec.

        The target module is fully initialized by its own import, so nothing is executed. Keeping its original
        spec preserves ``importlib.resources``, ``importlib.reload`` and relative imports in the target package.
        """
        module.__spec__ = self._target_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Map ``<alias>.<path>`` imports onto ``<target>.<path>``."""

    def __init__(self, alias: str, target: str) -> None:
        self.alias_prefix = alias + "."
        self.target_prefix = target + "."

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(self.alias_prefix):
            return None
        target_name = self.target_prefix + fullname[len(self.alias_prefix) :]
        if importlib.util.find_spec(target_name) is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(target_name))


def install_alias(alias: str, target: str) -> None:
    """Resolve every ``alias.<path>`` submodule import to ``target.<path>``; idempotent per alias."""
    if not any(isinstance(f, _AliasFinder) and f.alias_prefix == alias + "." for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder(alias, target))
