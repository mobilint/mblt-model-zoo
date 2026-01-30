import argparse
import os
from pathlib import Path

import pytest

from mblt_model_zoo.vision import ResNet50

TEST_DIR = Path(__file__).parent


@pytest.fixture
def resnet50():
    model = ResNet50()
    yield model
    model.dispose()


def run_inference(model, image_path, save_path):
    """Run inference with the given model and image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_img = model.preprocess(image_path)
    output = model(input_img)
    result = model.postprocess(output)

    result.plot(
        source_path=image_path,
        save_path=save_path,
        topk=5,
    )


def test_resnet50(resnet50):
    image_path = os.path.join(TEST_DIR, "rc", "volcano.jpg")
    save_path = os.path.join(
        TEST_DIR, "tmp", f"resnet50_{os.path.basename(image_path)}"
    )

    run_inference(resnet50, image_path, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResNet50 inference")
    parser.add_argument(
        "--mxq-path",
        type=str,
        default=None,
        help="Path to the ResNet50 model file (.mxq)",
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
    model = ResNet50(
        local_path=args.mxq_path,
        model_type=args.model_type,
        infer_mode=args.infer_mode,
        product=args.product,
    )
    if args.save_path is None:
        args.save_path = os.path.join(
            TEST_DIR, "tmp", f"resnet50_{os.path.basename(args.input_path)}"
        )

    try:
        run_inference(model, args.input_path, args.save_path)
    finally:
        model.dispose()
