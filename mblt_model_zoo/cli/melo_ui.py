"""Compatibility bridge for ``mblt-model-zoo melo-ui``: runs ``melotts-mblt ui``."""

from __future__ import annotations

import argparse

from ._melotts import exit_missing_dependency, load_standalone


def _cmd_melo_ui(args: argparse.Namespace) -> int:
    ui = load_standalone("cli.ui")
    if ui is None:
        return exit_missing_dependency()
    return ui.run_ui(share=args.share, host=args.host, port=args.port)


def add_melo_ui_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ``melo-ui`` with the same options as ``melotts-mblt ui``."""
    parser = subparsers.add_parser("melo-ui", help="Launch MeloTTS WebUI (Gradio; requires mblt-model-zoo[MeloTTS])")
    parser.add_argument(
        "--share",
        "-s",
        action="store_true",
        default=False,
        help="Expose a publicly-accessible shared Gradio link.",
    )
    parser.add_argument("--host", default=None, help="Server host / bind address (e.g., 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Server port (e.g., 7860)")
    parser.set_defaults(_handler=_cmd_melo_ui)
