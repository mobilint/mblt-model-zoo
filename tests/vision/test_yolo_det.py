"""
Test script for YOLO26m model.

This script tests the YOLO26m model by running inference on a sample image.
It can be run as a pytest test or as a standalone script.
"""

import argparse
import os
from pathlib import Path

import pytest

from mblt_model_zoo.vision import YOLO26m

TEST_DIR = Path(__file__).parent


@pytest.fixture
def yolo_det():
    """Fixture to initialize and dispose of the YOLO26m model."""
    model = YOLO26m()
    yield model
    model.dispose()


def run_inference(model, image_path, save_path, conf_thres=0.5, iou_thres=0.5):
    """Run inference with the given model and image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_img = model.preprocess(image_path)
    output = model(input_img)
    result = model.postprocess(output, conf_thres=conf_thres, iou_thres=iou_thres)

    result.plot(
        source_path=image_path,
        save_path=save_path,
    )


def test_yolo_det(yolo_det):
    """Test YOLO26m inference on a sample image."""
    image_path = os.path.join(TEST_DIR, "rc", "cr7.jpg")
    save_path = os.path.join(
        TEST_DIR,
        "tmp",
        f"yolo26m_{os.path.basename(image_path)}",
    )

    run_inference(yolo_det, image_path, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO26m inference")
    parser.add_argument(
        "--mxq-path",
        type=str,
        default=None,
        help="Path to the YOLO26m model file (.mxq)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="DEFAULT",
        help="Model type",
    )
    parser.add_argument(
        "--infer-mode",
        type=str,
        default="global",
        help="Inference mode",
    )
    parser.add_argument(
        "--product",
        type=str,
        default="aries",
        help="Product",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=os.path.join(TEST_DIR, "rc", "cr7.jpg"),
        help="Path to the input image",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the output image",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="IoU threshold",
    )

    args = parser.parse_args()

    # Load model with the specified mxq_path
    model = YOLO26m(
        local_path=args.mxq_path,
        model_type=args.model_type,
        infer_mode=args.infer_mode,
        product=args.product,
    )
    if args.save_path is None:
        args.save_path = os.path.join(
            TEST_DIR, "tmp", f"yolo26m_{os.path.basename(args.input_path)}"
        )

    try:
        run_inference(
            model, args.input_path, args.save_path, args.conf_thres, args.iou_thres
        )
    finally:
        model.dispose()
