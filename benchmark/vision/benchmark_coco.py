"""
Benchmark script for YOLOv5m on COCO dataset.

This script runs the benchmark for the YOLOv5m model using the COCO dataset.
It initializes the model and evaluates its performance.
"""

import argparse
import os

import mblt_model_zoo
from mblt_model_zoo.vision.utils.evaluation import eval_coco

TOTAL_MODELS = mblt_model_zoo.vision.list_models()
MODEL_LIST = (
    TOTAL_MODELS["object_detection"]
    + TOTAL_MODELS["instance_segmentation"]
    + TOTAL_MODELS["pose_estimation"]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- Model Configuration ---
    parser.add_argument(
        "--model-name",
        type=str,
        default="YOLOv5m",
        choices=MODEL_LIST,
        help="Model name",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to the YOLOv5m model file (.mxq)",
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
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/coco"),
        help="Path to the COCO data",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.001,
        help="Confidence threshold for object detection",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.7,
        help="IOU threshold for object detection",
    )
    args = parser.parse_args()

    model_cls = getattr(mblt_model_zoo.vision, args.model_name)
    # --- Model Initialization ---
    model = model_cls(args.local_path, args.model_type, args.infer_mode, args.product)

    # --- Benchmark Execution ---
    acc = eval_coco(
        model, args.data_path, args.batch_size, args.conf_thres, args.iou_thres
    )
    print(f"COCO evaluation completed. mAP 50:95: {acc:.5f}")
