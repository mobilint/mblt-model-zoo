"""Benchmark script for YOLO OBB models on DOTAv1."""

from __future__ import annotations

import argparse
import os

from mblt_model_zoo.vision import MBLT_Engine
from mblt_model_zoo.vision.utils.evaluation import eval_dota

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cls", type=str, default="YOLOv8s-obb", help="model class to evaluate")
    parser.add_argument("--model-path", type=str, default="", help="Path to a local MXQ or ONNX model file")
    parser.add_argument("--mxq-path", type=str, default="", help="Path to a local MXQ model file")
    parser.add_argument("--onnx-path", type=str, default="", help="Path to a local ONNX model file")
    parser.add_argument("--framework", type=str, default=None, choices=["mxq", "onnx"], help="Inference framework")
    parser.add_argument("--model-type", type=str, default="DEFAULT", help="model type")
    parser.add_argument(
        "--core-mode",
        type=str,
        default="global8",
        choices=["single", "multi", "global4", "global8"],
        help="Inference core mode",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/dotav1"),
        help="Path to the DOTAv1 data",
    )
    parser.add_argument("--conf-thres", type=float, default=0.01, help="Confidence threshold for OBB detection")
    parser.add_argument("--iou-thres", type=float, default=0.7, help="IoU threshold for OBB detection")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="benchmark/vision/results/dota",
        help="Directory for DOTA Task1 prediction files",
    )
    args = parser.parse_args()

    model = MBLT_Engine(
        model_cls=args.model_cls,
        model_path=args.model_path,
        mxq_path=args.mxq_path,
        onnx_path=args.onnx_path,
        model_type=args.model_type,
        core_mode=args.core_mode,
        framework=args.framework,
    )

    try:
        result = eval_dota(model, args.data_path, args.batch_size, args.conf_thres, args.iou_thres, args.save_dir)
        print(f"DOTAv1 evaluation completed. mAP test 50: {result.map50:.5f}, mAP test 50-95: {result.map5095:.5f}")
    finally:
        model.dispose()
