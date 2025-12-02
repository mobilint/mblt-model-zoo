import argparse
import os

from mblt_model_zoo.vision import ResNet50, eval_imagenet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResNet50 Benchmark")
    # --- Model Configuration ---
    parser.add_argument(
        "--mxq_path",
        type=str,
        default=None,
        help="Path to the ResNet50 model file (.mxq)",
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
        default=os.path.expanduser("~/.mblt_model_zoo/datasets/imagenet"),
        help="Path to the ImageNet data",
    )
    args = parser.parse_args()

    # --- Model Initialization ---
    model = ResNet50(args.mxq_path, args.model_type, args.infer_mode, args.product)

    # --- Benchmark Execution ---
    eval_imagenet(model, args.data_path, args.batch_size)
