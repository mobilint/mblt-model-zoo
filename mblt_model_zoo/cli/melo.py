from __future__ import annotations

import argparse
import sys

from transformers import HfArgumentParser


def _require_melotts_deps() -> None:
    try:
        import click  # noqa: F401
    except Exception as e:
        print(
            "Missing optional dependencies for MeloTTS CLI.\n"
            "Install with: pip install 'mblt-model-zoo[MeloTTS]'\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _cmd_melo(args: argparse.Namespace) -> int:
    _require_melotts_deps()

    from mblt_model_zoo.MeloTTS import main as melo_main

    try:
        melo_main.main(standalone_mode=False, args=list(args.melo_args))
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    return 0


def add_melo_parser(
    subparsers: argparse._SubParsersAction[HfArgumentParser],
) -> None:
    # Forward all args to the Click-based MeloTTS CLI so `mblt-model-zoo melo --help`
    # shows Click help (not argparse help).
    parser = subparsers.add_parser(
        "melo",
        aliases=["melotts"],
        add_help=False,
        help="MeloTTS CLI (alias: melotts)",
    )
    parser.add_argument("melo_args", nargs=argparse.REMAINDER)
    parser.set_defaults(_handler=_cmd_melo)
