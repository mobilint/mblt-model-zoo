"""
Benchmark script for ResNet50 on ImageNet dataset.

This script runs the benchmark for the ResNet50 model using the ImageNet dataset.
It initializes the model and evaluates its performance.
"""

import argparse
import os

import mblt_model_zoo
from mblt_model_zoo.vision.utils.evaluation import eval_imagenet

TOTAL_MODELS = mblt_model_zoo.vision.list_models()
MODEL_LIST = TOTAL_MODELS["image_classification"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- Model Configuration ---
    parser.add_argument(
        "--model-name",
        type=str,
        default="ResNet50",
        choices=MODEL_LIST,
        help="Model name",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to the ResNet50 model file (.mxq)",
    )
    parser.add_argument("--model-type", type=str, default="DEFAULT", help="Model type")
    parser.add_argument(
        "--infer-mode",
        type=str,
        default="global8",
        choices=["single", "multi", "global4", "global8"],
        help="Inference mode",
    )
    parser.add_argument("--product", type=str, default="aries", help="Product")
    # --- Benchmark Configuration ---
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/imagenet"),
        help="Path to the ImageNet data",
    )
    args = parser.parse_args()
    model_cls = getattr(mblt_model_zoo.vision, args.model_name)
    # --- Model Initialization ---
    model = model_cls(
        local_path=args.local_path,
        model_type=args.model_type,
        infer_mode=args.infer_mode,
        product=args.product,
    )

    # --- Benchmark Execution ---
    eval_imagenet(model, args.data_path, args.batch_size)
