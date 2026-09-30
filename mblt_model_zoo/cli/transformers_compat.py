"""Compatibility bridge for :mod:`transformers_mblt.cli.transformers_compat`.

Upstream Transformers commands (``chat``, ``serve``, ``download``, ...) are delegated by the standalone package.
When transformers-mblt is installed this module *is* ``transformers_mblt.cli.transformers_compat``; otherwise those
commands report how to install it instead of reaching argparse as unknown commands.
"""

from __future__ import annotations

import sys
from typing import Sequence

from ._transformers import exit_missing_dependency, load_standalone

_standalone = load_standalone("cli.transformers_compat")

if _standalone is not None:
    sys.modules[__name__] = _standalone
else:
    # Mirrors transformers_mblt.cli.transformers_compat.TRANSFORMERS_CLI_COMMANDS so these commands still get an
    # actionable install hint without the package.
    TRANSFORMERS_CLI_COMMANDS = frozenset(
        {
            "add-fast-image-processor",
            "add-new-model-like",
            "chat",
            "convert",
            "download",
            "env",
            "run",
            "serve",
            "version",
        }
    )

    def is_transformers_cli_command(argv: Sequence[str]) -> bool:
        """Return whether the argv targets an upstream Transformers CLI command."""
        return len(argv) > 1 and argv[1] in TRANSFORMERS_CLI_COMMANDS

    def dispatch_transformers_cli(argv: Sequence[str]) -> int:
        """Report that delegated Transformers commands need transformers-mblt."""
        return exit_missing_dependency()

    __all__ = ["TRANSFORMERS_CLI_COMMANDS", "dispatch_transformers_cli", "is_transformers_cli_command"]

    if __name__ == "__main__":
        raise SystemExit(exit_missing_dependency())
