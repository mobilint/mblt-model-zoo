"""Compatibility bridge for ``mblt-model-zoo melo`` / ``melotts``: runs ``melotts-mblt tts``."""

from __future__ import annotations

import argparse
from typing import Sequence

from ._melotts import exit_missing_dependency, load_standalone


def run_melo(args: Sequence[str], prog_name: str) -> int:
    """Run the standalone MeloTTS CLI, or report how to install it."""
    tts = load_standalone("cli.tts")
    if tts is None:
        return exit_missing_dependency()
    return tts.run_tts(args, prog_name=prog_name)


def _cmd_melo(args: argparse.Namespace) -> int:
    return run_melo(args.melo_args, prog_name="mblt-model-zoo melo")


def add_melo_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ``melo`` (alias ``melotts``); ``main()`` dispatches it before argparse so Click owns ``--help``."""
    parser = subparsers.add_parser(
        "melo",
        aliases=["melotts"],
        add_help=False,
        help="MeloTTS CLI (alias: melotts; requires mblt-model-zoo[MeloTTS])",
    )
    parser.add_argument("melo_args", nargs=argparse.REMAINDER)
    parser.set_defaults(_handler=_cmd_melo)
