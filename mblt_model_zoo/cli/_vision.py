"""Shared helpers for vision CLI commands."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path("runs") / "vision"


def parse_target_cores(value: str | None) -> list[str] | None:
    """Parses a semicolon-separated target core list."""

    if value is None:
        return None
    cores = [item.strip() for item in value.split(";") if item.strip()]
    return cores or None


def parse_target_clusters(value: str | None) -> list[int] | None:
    """Parses a semicolon-separated target cluster list."""

    if value is None:
        return None
    try:
        clusters = [int(item.strip()) for item in value.split(";") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target clusters must be semicolon-separated integers") from exc
    return clusters or None


def add_common_vision_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments shared by all vision inference commands."""

    parser.add_argument("--source", required=True, help="Path to the source image.")
    parser.add_argument("--model", required=True, help="Vision model name, for example `resnet50` or `yolo11m`.")
    parser.add_argument("--output", "--save-path", dest="output", help="Path to save the plotted result image.")
    parser.add_argument("--mxq-path", default="", help="Optional local MXQ model path.")
    parser.add_argument("--model-type", default="DEFAULT", help="Model variant from the YAML configuration.")
    parser.add_argument(
        "--core-mode",
        default="global8",
        choices=["single", "multi", "global4", "global8"],
        help="NPU core execution mode.",
    )
    parser.add_argument("--dev-no", type=int, default=0, help="NPU device number.")
    parser.add_argument(
        "--target-cores",
        type=parse_target_cores,
        help="Optional semicolon-separated core list for single-core mode, for example `0:0;0:1`.",
    )
    parser.add_argument(
        "--target-clusters",
        type=parse_target_clusters,
        help="Optional semicolon-separated cluster list for multi/global modes, for example `0;1`.",
    )


def add_threshold_args(parser: argparse.ArgumentParser) -> None:
    """Adds postprocess threshold arguments for dense vision tasks."""

    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold.")


def add_vision_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    command: str,
    help_text: str,
    handler: Any,
    aliases: list[str] | None = None,
) -> argparse.ArgumentParser:
    """Creates a vision command parser with common arguments."""

    parser = subparsers.add_parser(command, aliases=aliases or [], help=help_text)
    parser.set_defaults(_handler=handler)
    add_common_vision_args(parser)
    return parser


def build_default_output_path(command: str, source: str, model: str) -> str:
    """Builds the default path used for plotted vision command output."""

    source_path = Path(source)
    suffix = source_path.suffix or ".jpg"
    return str(DEFAULT_OUTPUT_DIR / command / f"{source_path.stem}_{model}{suffix}")


def resolve_output_path(output: str | None, command: str, source: str, model: str) -> str:
    """Returns an absolute result image path and ensures its parent exists."""

    save_path = Path(output or build_default_output_path(command, source, model)).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return str(save_path.resolve())


def require_source_file(source: str) -> None:
    """Exits with a clear message when the source image is unavailable."""

    source_path = Path(source).expanduser()
    if not source_path.is_file():
        raise SystemExit(f"Source image not found: {source}")


def run_vision_inference(
    args: argparse.Namespace,
    *,
    command: str,
) -> Any:
    """Runs a complete vision inference pipeline for a CLI command."""

    require_source_file(args.source)

    try:
        from mblt_model_zoo.vision import MBLT_Engine
    except ImportError as exc:
        print(f"Missing dependencies for vision CLI: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    model = MBLT_Engine(
        model_cls=args.model,
        model_type=args.model_type,
        mxq_path=args.mxq_path,
        dev_no=args.dev_no,
        core_mode=args.core_mode,
        target_cores=args.target_cores,
        target_clusters=args.target_clusters,
    )
    try:
        actual_task = str(model.post_cfg.get("task", "")).lower()
        postprocess_kwargs: dict[str, Any] = {}
        plot_kwargs: dict[str, Any] = {}
        if actual_task == "image_classification":
            plot_kwargs["topk"] = args.topk
        elif actual_task in {"object_detection", "instance_segmentation", "pose_estimation"}:
            postprocess_kwargs = {"conf_thres": args.conf_thres, "iou_thres": args.iou_thres}

        input_img = model.preprocess(args.source)
        output = model(input_img)
        result = model.postprocess(output, **postprocess_kwargs)

        save_path = resolve_output_path(args.output, command, args.source, args.model)
        result.plot(source_path=args.source, save_path=save_path, **plot_kwargs)
        print(f"Saved result to {os.path.relpath(save_path)}")
        return result
    finally:
        model.dispose()
