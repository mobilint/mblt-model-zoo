from __future__ import annotations

import argparse
import sys

from transformers import HfArgumentParser


def _require_melotts_deps() -> None:
    try:
        import gradio  # noqa: F401
    except Exception as e:
        print(
            "Missing optional dependencies for MeloTTS WebUI.\n"
            "Install with: pip install 'mblt-model-zoo[MeloTTS]'\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _cmd_melo_ui(args: argparse.Namespace) -> int:
    _require_melotts_deps()

    from mblt_model_zoo.MeloTTS import app as melo_app

    click_args: list[str] = []
    if args.share:
        click_args.append("--share")
    if args.host is not None:
        click_args.extend(["--host", args.host])
    if args.port is not None:
        click_args.extend(["--port", str(args.port)])

    try:
        melo_app.main(standalone_mode=False, args=click_args)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    return 0


def add_melo_ui_parser(
    subparsers: argparse._SubParsersAction[HfArgumentParser],
) -> None:
    parser = subparsers.add_parser("melo-ui", help="Launch MeloTTS WebUI (Gradio)")
    parser.add_argument(
        "--share",
        "-s",
        action="store_true",
        default=False,
        help="Expose a publicly-accessible shared Gradio link.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Server host / bind address (e.g., 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (e.g., 7860)",
    )
    parser.set_defaults(_handler=_cmd_melo_ui)
