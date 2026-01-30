import argparse
import os

from mblt_model_zoo.vision import ResNet50
from mblt_model_zoo.vision.utils.evaluation import eval_imagenet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResNet50 Benchmark")
    # --- Model Configuration ---
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to the ResNet50 model file (.mxq)",
    )
    parser.add_argument("--model-type", type=str, default="DEFAULT", help="Model type")
    parser.add_argument(
        "--infer-mode", type=str, default="global", help="Inference mode"
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

    # --- Model Initialization ---
    model = ResNet50(
        local_path=args.local_path,
        model_type=args.model_type,
        infer_mode=args.infer_mode,
        product=args.product,
    )

    # --- Benchmark Execution ---
    eval_imagenet(model, args.data_path, args.batch_size)
