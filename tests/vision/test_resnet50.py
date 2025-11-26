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


def test_resnet50(resnet50):
    image_path = os.path.join(TEST_DIR, "rc", "volcano.jpg")
    save_path = os.path.join(
        TEST_DIR, "tmp", f"resnet50_{os.path.basename(image_path)}"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # resnet50.gpu()
    input_img = resnet50.preprocess(image_path)
    output = resnet50(input_img)
    result = resnet50.postprocess(output)

    result.plot(
        source_path=image_path,
        save_path=save_path,
        topk=5,
    )
