from __future__ import annotations

from argparse import ArgumentParser
from typing import cast

import transformers
from transformers import HfArgumentParser
from transformers.commands.chat import ChatCommand

from .chat import register_mobilint_models
from .melo import add_melo_parser
from .melo_ui import add_melo_ui_parser
from .tps import add_tps_parser


def build_parser() -> HfArgumentParser:
    parser = HfArgumentParser(prog="mblt-model-zoo")
    commands_parser = parser.add_subparsers(help="mblt-model-zoo command helpers")

    add_tps_parser(commands_parser)
    add_melo_parser(commands_parser)
    add_melo_ui_parser(commands_parser)
    ChatCommand.register_subcommand(cast(ArgumentParser, commands_parser))

    return parser


def main():
    # Click-based MeloTTS CLI needs to accept arbitrary options/args (including `--help`)
    # without argparse rejecting them, so we delegate early.
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in {"melo", "melotts"}:
        from .melo import _require_melotts_deps

        _require_melotts_deps()
        from mblt_model_zoo.MeloTTS import main as melo_main

        try:
            melo_main.main(
                standalone_mode=False,
                prog_name=f"{sys.argv[0]} {sys.argv[1]}",
                args=sys.argv[2:],
            )
        except SystemExit as e:
            exit(int(e.code) if e.code is not None else 0)
        exit(0)

    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "func"):
        register_mobilint_models(args, transformers)
        service = args.func(args)
        service.run()
        exit(0)

    if hasattr(args, "_handler"):
        exit(args._handler(args))

    parser.print_help()
    exit(1)


if __name__ == "__main__":
    main()
