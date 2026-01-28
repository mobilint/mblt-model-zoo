from __future__ import annotations

from argparse import ArgumentParser
from typing import cast

import transformers
from transformers import HfArgumentParser
from transformers.commands.chat import ChatCommand

from .chat import register_mobilint_models
from .tps import add_tps_parser


def build_parser() -> HfArgumentParser:
    parser = HfArgumentParser(prog="mblt-model-zoo")
    commands_parser = parser.add_subparsers(help="mblt-model-zoo command helpers")

    add_tps_parser(commands_parser)
    ChatCommand.register_subcommand(cast(ArgumentParser, commands_parser))

    return parser


def main():
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
