from __future__ import annotations

import argparse

from .compile import add_compile_parser
from .melo import add_melo_parser
from .melo_ui import add_melo_ui_parser
from .predict import add_predict_parser
from .tps import add_tps_parser
from .transformers_compat import dispatch_transformers_cli, is_transformers_cli_command
from .val import add_val_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mblt-model-zoo",
        description=(
            "Mobilint CLI helpers. Upstream Transformers commands such as "
            "`chat`, `serve`, `download`, `env`, and `version` are delegated "
            "to the installed `transformers` package."
        ),
    )
    commands_parser = parser.add_subparsers(help="mblt-model-zoo command helpers")

    add_predict_parser(commands_parser)
    add_val_parser(commands_parser)
    add_compile_parser(commands_parser)
    add_tps_parser(commands_parser)
    add_melo_parser(commands_parser)
    add_melo_ui_parser(commands_parser)

    return parser


def main():
    # Click-based MeloTTS CLI needs to accept arbitrary options/args (including `--help`)
    # without argparse rejecting them, so we delegate early.
    import sys

    if is_transformers_cli_command(sys.argv):
        return dispatch_transformers_cli(sys.argv)

    if len(sys.argv) > 1 and sys.argv[1] in {"melo", "melotts"}:
        from .melo import run_melo

        return run_melo(sys.argv[2:], prog_name=f"{sys.argv[0]} {sys.argv[1]}")

    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "_handler"):
        return args._handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
