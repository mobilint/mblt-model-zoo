"""
Benchmark script for ResNet50 on ImageNet dataset.

This script runs the benchmark for the ResNet50 model using the ImageNet dataset.
It initializes the model and evaluates its performance.
"""

import argparse
import os

import mblt_model_zoo
from mblt_model_zoo.vision.utils.evaluation import eval_imagenet

MODEL_LIST = [
    "AlexNet",
    "ConvNeXt_Base",
    "ConvNeXt_Large",
    "ConvNeXt_Small",
    "ConvNeXt_Tiny",
    "DeiT3_Base_Patch16_224",
    "DeiT3_Base_Patch16_384",
    "DeiT3_Large_Patch16_224",
    "DeiT3_Large_Patch16_384",
    "DeiT3_Medium_Patch16_224",
    "DeiT3_Small_Patch16_224",
    "DeiT3_Small_Patch16_384",
    "DeiT_Base_Patch16_224",
    "DeiT_Base_Patch16_384",
    "DeiT_Small_Patch16_224",
    "DeiT_Tiny_Patch16_224",
    "DenseNet121",
    "DenseNet169",
    "DenseNet201",
    "FlexiViT_Base",
    "FlexiViT_Large",
    "FlexiViT_Small",
    "GoogLeNet",
    "Inception_V3",
    "MNasNet1_0",
    "MNasNet1_3",
    "MobileNet_V2",
    "RegNet_X_16GF",
    "RegNet_X_1_6GF",
    "RegNet_X_32GF",
    "RegNet_X_3_2GF",
    "RegNet_X_400MF",
    "RegNet_X_800MF",
    "RegNet_X_8GF",
    "RegNet_Y_16GF",
    "RegNet_Y_1_6GF",
    "RegNet_Y_32GF",
    "RegNet_Y_3_2GF",
    "RegNet_Y_400MF",
    "RegNet_Y_800MF",
    "RegNet_Y_8GF",
    "ResNet101",
    "ResNet152",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNext101_32x8d",
    "ResNext101_64x4d",
    "ResNext50_32x4d",
    "ShuffleNet_V2_X1_0",
    "ShuffleNet_V2_X1_5",
    "ShuffleNet_V2_X2_0",
    "VGG11",
    "VGG11_BN",
    "VGG13",
    "VGG13_BN",
    "VGG16",
    "VGG16_BN",
    "VGG19",
    "VGG19_BN",
    "ViT_Base_Patch16_224",
    "ViT_Base_Patch16_384",
    "ViT_Base_Patch32_224",
    "ViT_Base_Patch32_384",
    "ViT_Base_Patch8_224",
    "ViT_Large_Patch16_224",
    "ViT_Large_Patch16_384",
    "ViT_Large_Patch32_384",
    "ViT_Small_Patch16_224",
    "ViT_Small_Patch16_384",
    "ViT_Small_Patch32_224",
    "ViT_Small_Patch32_384",
    "ViT_Tiny_Patch16_224",
    "ViT_Tiny_Patch16_384",
    "Wide_ResNet101_2",
    "Wide_ResNet50_2",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResNet50 Benchmark")
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
        default="global",
        choices=["global", "single", "multi", "global4", "global8"],
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
