import os
from pathlib import Path

import pytest

from mblt_model_zoo.vision import YOLOv8lSeg

TEST_DIR = Path(__file__).parent


@pytest.fixture
def yolo_seg():
    model = YOLOv8lSeg()
    yield model
    model.dispose()


def test_yolo_seg(yolo_seg):
    image_path = os.path.join(TEST_DIR, "rc", "cr7.jpg")
    save_path = os.path.join(
        TEST_DIR,
        "tmp",
        f"yolov8l_seg_{os.path.basename(image_path)}",
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_img = yolo_seg.preprocess(image_path)
    output = yolo_seg(input_img)
    result = yolo_seg.postprocess(output, conf_thres=0.5, iou_thres=0.5)

    result.plot(
        source_path=image_path,
        save_path=save_path,
    )
