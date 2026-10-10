from __future__ import annotations

import argparse
from typing import Sequence

from .._deprecation import print_cli_notice
from .compile import add_compile_parser
from .melo import add_melo_parser
from .melo_ui import add_melo_ui_parser
from .predict import add_predict_parser
from .tps import add_tps_parser
from .transformers_compat import dispatch_transformers_cli, is_transformers_cli_command
from .val import add_val_parser

# Legacy subcommand -> (replacement distribution, replacement command prefix).
_REPLACEMENTS: dict[str, tuple[str, str]] = {
    "predict": ("mblt-vision-python", "mblt-vision predict"),
    "val": ("mblt-vision-python", "mblt-vision val"),
    "compile": ("mblt-vision-python", "mblt-vision compile"),
    "tps": ("transformers-mblt", "transformers-mblt tps"),
    "melo": ("melotts-mblt", "melotts-mblt tts"),
    "melotts": ("melotts-mblt", "melotts-mblt tts"),
    "melo-ui": ("melotts-mblt", "melotts-mblt ui"),
}


def _print_deprecation_notice(argv: Sequence[str]) -> None:
    """Print the deprecation notice naming the standalone command that replaces this invocation."""
    command = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else None
    if command is None:
        print_cli_notice(None, None, None)
    elif command in _REPLACEMENTS:
        package, replacement = _REPLACEMENTS[command]
        print_cli_notice(f"mblt-model-zoo {command}", package, replacement)
    elif is_transformers_cli_command(argv):
        print_cli_notice(f"mblt-model-zoo {command}", "transformers-mblt", f"transformers-mblt {command}")
    else:
        print_cli_notice(None, None, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mblt-model-zoo",
        description=(
            "DEPRECATED: mblt-model-zoo no longer receives updates. Use `mblt-vision` (mblt-vision-python), "
            "`transformers-mblt`, and `melotts-mblt` instead. "
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

    _print_deprecation_notice(sys.argv)

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
