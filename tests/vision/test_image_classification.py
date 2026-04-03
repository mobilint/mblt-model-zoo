"""
Test script for ResNet50 model.

This script tests the ResNet50 model by running inference on a sample image.
It can be run as a pytest test or as a standalone script.
"""

import argparse
import os
from pathlib import Path

import pytest

from mblt_model_zoo.vision import MBLT_Engine

TEST_DIR = Path(__file__).parent


@pytest.fixture
def pytest_model():
    """Fixture to initialize and dispose of the ResNet50 model."""
    model = MBLT_Engine(
        model_cls="resnet50",
        mxq_path="",
        model_type="DEFAULT",
        core_mode="global8",
    )
    yield model
    model.dispose()


def run_inference(
    model,
    image_path,
    save_path,
):
    """Run inference with the given model and image.

    Args:
            model: The vision model to use.
            image_path: Path to the input image.
            save_path: Path to save the output visualization.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_img = model.preprocess(image_path)
    output = model(input_img)
    result = model.postprocess(output)

    result.plot(
        source_path=image_path,
        save_path=save_path,
        topk=5,
    )


def test_image_classification(pytest_model):
    """Test ResNet50 inference on a sample image."""
    image_path = os.path.join(TEST_DIR, "rc", "volcano.jpg")
    save_path = os.path.join(TEST_DIR, "tmp", f"resnet50_{os.path.basename(image_path)}")

    run_inference(pytest_model, image_path, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image classification")
    parser.add_argument("--model-cls", type=str, default="resnet50", help="model type you want to test")
    parser.add_argument(
        "--mxq-path",
        type=str,
        default="",
        help="Path to the ResNet50 model file (.mxq)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="DEFAULT",
        help="Model type",
    )
    parser.add_argument(
        "--core-mode",
        type=str,
        default="global8",
        choices=["single", "multi", "global4", "global8"],
        help="Inference core mode",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=os.path.join(TEST_DIR, "rc", "volcano.jpg"),
        help="Path to the input image",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the output image",
    )

    args = parser.parse_args()

    # Load model with the specified mxq_path
    model = MBLT_Engine(
        model_cls=args.model_cls,
        mxq_path=args.mxq_path,
        model_type=args.model_type,
        core_mode=args.core_mode,
    )
    if args.save_path is None:
        args.save_path = os.path.join(TEST_DIR, "tmp", f"resnet50_{os.path.basename(args.input_path)}")

    try:
        run_inference(model, args.input_path, args.save_path)
    finally:
        model.dispose()
