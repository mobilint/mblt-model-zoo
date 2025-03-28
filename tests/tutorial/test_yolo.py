from argparse import ArgumentParser
import numpy as np
from mblt_models.vision import YOLOv5m, YOLOv8m

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="/workspace/mblt-models/tests/rc/cr7.jpg"
    )
    args = parser.parse_args()
    image_path = args.image_path

    yolov8m = YOLOv8m(local_model="/workspace/mblt-models/tmp/yolov8m.mxq")
    yolov8m_pre = yolov8m.get_preprocess()
    yolov8m_post = yolov8m.get_postprocess(conf_thres=0.5, iou_thres=0.5)

    input = yolov8m_pre(image_path)
    output = yolov8m(input)
    result = yolov8m_post(output)
    print(result)
