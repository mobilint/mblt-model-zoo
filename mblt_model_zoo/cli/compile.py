"""Vision compilation CLI command."""

from __future__ import annotations

import argparse
import sys


def _run_compile(args: argparse.Namespace) -> int:
    """Compile a vision model from parsed CLI arguments.

    Args:
        args: Parsed compilation arguments.

    Returns:
        Successful process status.
    """

    try:
        from mblt_model_zoo.compile.vision import compile_vision_model

        output_path = compile_vision_model(
            model_cls=args.model_cls,
            model_type=args.model_type,
            model_path=args.model_path,
            data_path=args.data_path,
            subset_path=args.subset_path,
            calib_data_path=args.calib_data_path,
            save_path=args.save_path,
            subset_size=args.subset_size,
            seed=args.seed,
            percentile=args.percentile,
            topk_ratio=args.topk_ratio,
        )
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"Compiled MXQ model to {output_path}")
    return 0


def add_compile_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Register the vision compilation command.

    Args:
        subparsers: Main CLI subparser collection.

    Returns:
        Registered compile command parser.
    """

    parser = subparsers.add_parser("compile", help="Compile a configured vision ONNX model to MXQ.")
    parser.set_defaults(_handler=_run_compile)
    parser.add_argument("--model-cls", required=True, help="Vision model name, for example `alexnet` or `yolo11m`.")
    parser.add_argument("--model-type", default="DEFAULT", help="Model variant from the YAML configuration.")
    parser.add_argument(
        "--model-path",
        "--onnx-path",
        dest="model_path",
        help="Preferred local ONNX file. Missing files fall back to the configured hosted artifact.",
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-path",
        help="Original organized dataset root; organize, sample, and preprocess it as needed.",
    )
    data_group.add_argument(
        "--subset-path",
        help="Already-sampled image subset; skip original dataset preparation and sampling.",
    )
    data_group.add_argument(
        "--calib-data-path",
        "--calib-data-dir",
        dest="calib_data_path",
        help="Ready directory of preprocessed .npy tensors; pass it directly to qbcompiler.",
    )
    parser.add_argument(
        "--save-path",
        help="Output MXQ path. Defaults to the ONNX stem under ~/.mblt_model_zoo.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        help="Per-category ImageNet/WiderFace count or total count for other datasets.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic calibration subset seed.")
    parser.add_argument("--percentile", type=float, help="Quantization percentile override.")
    parser.add_argument("--topk-ratio", type=float, help="Quantization top-k ratio override.")
    return parser


__all__ = ["add_compile_parser"]
