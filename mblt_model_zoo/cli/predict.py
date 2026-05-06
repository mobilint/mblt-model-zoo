"""Vision prediction CLI command."""

from __future__ import annotations

import argparse

from transformers import HfArgumentParser

from ._vision import add_threshold_args, add_vision_parser, run_vision_inference


def _cmd_predict(args: argparse.Namespace) -> int:
    """Runs vision inference on a source image."""

    run_vision_inference(args, command="predict")
    return 0


def add_predict_parser(
    subparsers: argparse._SubParsersAction[HfArgumentParser],
) -> None:
    """Registers the unified vision prediction CLI command."""

    parser = add_vision_parser(
        subparsers,
        command="predict",
        aliases=["classify", "detect", "pose", "segment"],
        help_text="Run vision inference on an image.",
        handler=_cmd_predict,
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of classification labels to show.")
    add_threshold_args(parser)
