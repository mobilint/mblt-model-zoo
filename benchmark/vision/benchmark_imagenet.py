"""
Benchmark script for ResNet50 on ImageNet dataset.

This script runs the benchmark for the ResNet50 model using the ImageNet dataset.
It initializes the model and evaluates its performance.
"""

import argparse
import os

from mblt_model_zoo.vision import MBLT_Engine
from mblt_model_zoo.vision.utils.evaluation import eval_imagenet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- Model Configuration ---
    parser.add_argument("--model-cls", type=str, default="resnet50", help="model type you want to test")
    parser.add_argument(
        "--mxq-path",
        type=str,
        default="",
        help="Path to the ResNet50 model file (.mxq)",
    )
    parser.add_argument("--model-type", type=str, default="DEFAULT", help="model type")
    parser.add_argument(
        "--core-mode",
        type=str,
        default="global8",
        choices=["single", "multi", "global4", "global8"],
        help="Inference core mode",
    )
    # --- Benchmark Configuration ---
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/imagenet"),
        help="Path to the ImageNet data",
    )
    args = parser.parse_args()
    # Load model with the specified mxq_path
    model = MBLT_Engine(
        model_cls=args.model_cls,
        mxq_path=args.mxq_path,
        model_type=args.model_type,
        core_mode=args.core_mode,
    )

    # --- Benchmark Execution ---
    eval_imagenet(model, args.data_path, args.batch_size)
