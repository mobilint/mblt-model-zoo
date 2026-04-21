from __future__ import annotations

from transformers import HfArgumentParser

from .melo import add_melo_parser
from .melo_ui import add_melo_ui_parser
from .tps import add_tps_parser
from .transformers_compat import dispatch_transformers_cli, is_transformers_cli_command


def build_parser() -> HfArgumentParser:
    parser = HfArgumentParser(
        prog="mblt-model-zoo",
        description=(
            "Mobilint CLI helpers. Upstream Transformers commands such as "
            "`chat`, `serve`, `download`, `env`, and `version` are delegated "
            "to the installed `transformers` package."
        ),
    )
    commands_parser = parser.add_subparsers(help="mblt-model-zoo command helpers")

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

    if hasattr(args, "_handler"):
        return args._handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
