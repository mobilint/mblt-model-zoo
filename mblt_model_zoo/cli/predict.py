"""Vision prediction CLI command."""

from __future__ import annotations

import argparse

from ._vision import add_e2e_arg, add_threshold_args, add_vision_parser, run_vision_inference


def _cmd_predict(args: argparse.Namespace) -> int:
    """Runs vision inference on a source image."""

    run_vision_inference(args, command="predict")
    return 0


def add_predict_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
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
    parser.add_argument(
        "--raw-output",
        help="Path to save raw export-style output with `--e2e false`.",
    )
    add_threshold_args(parser, conf_default=0.25, iou_default=None)
    add_e2e_arg(parser)
