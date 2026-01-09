import argparse
import os

from mblt_model_zoo.vision import YOLO11m, eval_coco

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO11m Benchmark")
    # --- Model Configuration ---
    parser.add_argument(
        "--local_path",
        type=str,
        default=None,
        help="Path to the YOLO11m model file (.mxq)",
    )
    parser.add_argument("--model_type", type=str, default="DEFAULT", help="Model type")
    parser.add_argument(
        "--infer_mode", type=str, default="global", help="Inference mode"
    )
    parser.add_argument("--product", type=str, default="aries", help="Product")
    # --- Benchmark Configuration ---
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/coco"),
        help="Path to the COCO data",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.001,
        help="Confidence threshold for object detection",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.7,
        help="IOU threshold for object detection",
    )
    args = parser.parse_args()

    # --- Model Initialization ---
    model = YOLO11m(args.local_path, args.model_type, args.infer_mode, args.product)

    # --- Benchmark Execution ---
    acc = eval_coco(
        model, args.data_path, args.batch_size, args.conf_thres, args.iou_thres
    )
    print(f"COCO evaluation completed. mAP 50:95: {acc:.5f}")
