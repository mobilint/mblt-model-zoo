import os
from argparse import ArgumentParser
from mblt_model_zoo.vision import YOLOv7e6e

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="/workspace/mblt-model-zoo/tests/rc/cr7.jpg"
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
            "/workspace/mblt-model-zoo/tests/tmp/",
            "yolo7e6e_" + os.path.basename(image_path),
        )

    yolo = YOLOv7e6e()

    input_img = yolo.preprocess(image_path)
    output = yolo(input_img)
    result = yolo.postprocess(output, conf_thres=0.5, iou_thres=0.5)

    result.plot(
        source_path=image_path,
        save_path=save_path,
    )
