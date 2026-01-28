from __future__ import annotations

import argparse

import transformers
from transformers import HfArgumentParser
from transformers.commands.chat import ChatCommand

from .tps import add_tps_parser
from .chat import register_mobilint_models

def build_parser() -> argparse.ArgumentParser:
    parser = HfArgumentParser(prog="mblt-model-zoo")
    commands_parser = parser.add_subparsers(help="mblt-model-zoo command helpers")

    add_tps_parser(commands_parser)
    ChatCommand.register_subcommand(commands_parser)

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
