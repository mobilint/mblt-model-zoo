import os
from pathlib import Path

import pytest

from mblt_model_zoo.vision import YOLOv9c

TEST_DIR = Path(__file__).parent


@pytest.fixture
def yolo_det():
    model = YOLOv9c()
    yield model
    model.dispose()


def test_yolo_det(yolo_det):
    image_path = os.path.join(TEST_DIR, "rc", "cr7.jpg")
    save_path = os.path.join(
        TEST_DIR,
        "tmp",
        f"yolov9c_{os.path.basename(image_path)}",
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_img = yolo_det.preprocess(image_path)
    output = yolo_det(input_img)
    result = yolo_det.postprocess(output, conf_thres=0.5, iou_thres=0.5)

    result.plot(
        source_path=image_path,
        save_path=save_path,
    )
