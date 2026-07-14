"""Compatibility entry point for packaged vision compilation."""

from __future__ import annotations

import argparse

from mblt_model_zoo.compile.vision import compile_vision_model


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone compatibility parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Compile a configured vision ONNX model to MXQ")
    parser.add_argument("--model-cls", required=True, help="Vision model name.")
    parser.add_argument("--model-type", default="DEFAULT", help="Model variant from the YAML configuration.")
    parser.add_argument("--model-path", "--onnx-path", dest="model_path", help="Preferred local ONNX file.")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--data-path", help="Original organized dataset root.")
    data_group.add_argument("--subset-path", help="Already-sampled image subset root.")
    data_group.add_argument(
        "--calib-data-path",
        "--calib-data-dir",
        dest="calib_data_path",
        help="Ready directory of preprocessed .npy tensors.",
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
    parser.add_argument("--seed", type=int, default=0, help="Calibration selection seed.")
    parser.add_argument("--percentile", type=float, help="Quantization percentile override.")
    parser.add_argument("--topk-ratio", type=float, help="Quantization top-k ratio override.")
    return parser


def main() -> None:
    """Run standalone vision compilation."""

    args = build_parser().parse_args()
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
    print(f"Compiled MXQ model to {output_path}")


if __name__ == "__main__":
    main()
