"""
Benchmark script for YOLOv5m on COCO dataset.

This script runs the benchmark for the YOLOv5m model using the COCO dataset.
It initializes the model and evaluates its performance.
"""

import argparse
import os

import mblt_model_zoo
from mblt_model_zoo.vision.utils.evaluation import eval_coco

MODEL_LIST = (
    [
        "YOLO11lSeg",
        "YOLO11mSeg",
        "YOLO11nSeg",
        "YOLO11sSeg",
        "YOLO11xSeg",
        "YOLO26lSeg",
        "YOLO26mSeg",
        "YOLO26nSeg",
        "YOLO26sSeg",
        "YOLO26xSeg",
        "YOLOv5lSeg",
        "YOLOv5mSeg",
        "YOLOv5nSeg",
        "YOLOv5sSeg",
        "YOLOv5xSeg",
        "YOLOv8lSeg",
        "YOLOv8mSeg",
        "YOLOv8nSeg",
        "YOLOv8sSeg",
        "YOLOv8xSeg",
        "YOLOv9cSeg",
        "YOLOv9eSeg",
    ]
    + [
        "GELANc",
        "GELANe",
        "GELANm",
        "GELANs",
        "YOLO11l",
        "YOLO11m",
        "YOLO11n",
        "YOLO11s",
        "YOLO11x",
        "YOLO12l",
        "YOLO12m",
        "YOLO12n",
        "YOLO12s",
        "YOLO12x",
        "YOLO26l",
        "YOLO26m",
        "YOLO26n",
        "YOLO26s",
        "YOLO26x",
        "YOLOv10b",
        "YOLOv10l",
        "YOLOv10m",
        "YOLOv10n",
        "YOLOv10s",
        "YOLOv10x",
        "YOLOv3_spp",
        "YOLOv3_sppu",
        "YOLOv3u",
        "YOLOv5l",
        "YOLOv5l6",
        "YOLOv5l6u",
        "YOLOv5lu",
        "YOLOv5m",
        "YOLOv5m6",
        "YOLOv5m6u",
        "YOLOv5mu",
        "YOLOv5n",
        "YOLOv5n6",
        "YOLOv5n6u",
        "YOLOv5nu",
        "YOLOv5s",
        "YOLOv5s6",
        "YOLOv5s6u",
        "YOLOv5su",
        "YOLOv5x",
        "YOLOv5x6",
        "YOLOv5x6u",
        "YOLOv5xu",
        "YOLOv7",
        "YOLOv7d6",
        "YOLOv7e6",
        "YOLOv7e6e",
        "YOLOv7w6",
        "YOLOv7x",
        "YOLOv8l",
        "YOLOv8m",
        "YOLOv8n",
        "YOLOv8s",
        "YOLOv8x",
        "YOLOv9c",
        "YOLOv9e",
        "YOLOv9m",
        "YOLOv9s",
        "YOLOv9t",
    ]
    + [
        "YOLO11lPose",
        "YOLO11mPose",
        "YOLO11nPose",
        "YOLO11sPose",
        "YOLO11xPose",
        "YOLO26lPose",
        "YOLO26mPose",
        "YOLO26nPose",
        "YOLO26sPose",
        "YOLO26xPose",
        "YOLOv8lPose",
        "YOLOv8mPose",
        "YOLOv8nPose",
        "YOLOv8sPose",
        "YOLOv8xPose",
    ]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv5m Benchmark")
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
        choices=["global", "single", "multi", "global4", "global8"],
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
