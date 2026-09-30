"""Compatibility bridge for :mod:`transformers_mblt.cli.tps`.

``mblt-model-zoo tps`` runs the standalone ``transformers-mblt tps`` implementation. When transformers-mblt is
installed this module *is* ``transformers_mblt.cli.tps``; otherwise ``tps`` reports how to install it.
"""

from __future__ import annotations

import argparse
import sys

from ._transformers import exit_missing_dependency, load_standalone

_standalone = load_standalone("cli.tps")

if _standalone is not None:
    sys.modules[__name__] = _standalone
else:

    def add_tps_parser(subparsers: argparse._SubParsersAction) -> None:
        """Register a ``tps`` placeholder that explains how to install transformers-mblt."""
        parser = subparsers.add_parser(
            "tps", help="Measure/sweep tokens-per-second (requires mblt-model-zoo[transformers])"
        )
        parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
        parser.set_defaults(_handler=lambda args: exit_missing_dependency())

    __all__ = ["add_tps_parser"]

    if __name__ == "__main__":
        raise SystemExit(exit_missing_dependency())
