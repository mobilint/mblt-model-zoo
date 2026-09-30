"""Compatibility bridge for :mod:`transformers_mblt.cli.chat` (requires mblt-model-zoo[transformers]).

When transformers-mblt is installed this module *is* ``transformers_mblt.cli.chat``. Otherwise it stays importable,
using any of its names raises ``ModuleNotFoundError`` with the install hint, and running it exits with status 2.
"""

import sys

from ._transformers import exit_missing_dependency, load_standalone, missing_dependency_getattr

_standalone = load_standalone("cli.chat")

if _standalone is not None:
    sys.modules[__name__] = _standalone
else:
    __getattr__ = missing_dependency_getattr(__name__)
    if __name__ == "__main__":
        raise SystemExit(exit_missing_dependency())
