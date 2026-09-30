"""Compatibility alias for :mod:`transformers_mblt.cli.chat` (requires mblt-model-zoo[transformers])."""

import sys

from ._transformers import INSTALL_HINT, load_standalone

_standalone = load_standalone("cli.chat")
if _standalone is None:
    raise ModuleNotFoundError(INSTALL_HINT, name="transformers_mblt")
sys.modules[__name__] = _standalone
