from __future__ import annotations

from transformers import HfArgumentParser

from .tps import add_tps_parser


def build_parser() -> HfArgumentParser:
    parser = HfArgumentParser(prog="mblt-model-zoo")
    commands_parser = parser.add_subparsers(help="mblt-model-zoo command helpers")

    add_tps_parser(commands_parser)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "_handler"):
        exit(args._handler(args))

    parser.print_help()
    exit(1)


if __name__ == "__main__":
    main()
