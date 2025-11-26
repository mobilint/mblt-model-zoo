import os
from pathlib import Path

import pytest

from mblt_model_zoo.vision import YOLO11xPose

TEST_DIR = Path(__file__).parent


@pytest.fixture
def yolo_pose():
    model = YOLO11xPose()
    yield model
    model.dispose()


def test_yolo_pose(yolo_pose):
    image_path = os.path.join(TEST_DIR, "rc", "cr7.jpg")
    save_path = os.path.join(
        TEST_DIR,
        "tmp",
        f"yolov11x_pose_{os.path.basename(image_path)}",
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    input_img = yolo_pose.preprocess(image_path)
    output = yolo_pose(input_img)
    result = yolo_pose.postprocess(output, conf_thres=0.5, iou_thres=0.5)

    result.plot(
        source_path=image_path,
        save_path=save_path,
    )
