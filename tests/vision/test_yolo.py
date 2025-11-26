import os
from argparse import ArgumentParser
from pathlib import Path

from mblt_model_zoo.vision import YOLOv9c

TEST_DIR = Path(__file__).parent

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default=os.path.join(TEST_DIR, "rc", "cr7.jpg")
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    image_path = args.image_path
    if args.save_path is not None:
        save_path = args.save_path
    else:
        save_path = os.path.join(
            TEST_DIR,
            "tmp",
            f"yolov9c_{os.path.basename(image_path)}",
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    yolo = YOLOv9c()

    input_img = yolo.preprocess(image_path)
    output = yolo(input_img)
    result = yolo.postprocess(output, conf_thres=0.5, iou_thres=0.5)

    result.plot(
        source_path=image_path,
        save_path=save_path,
    )
