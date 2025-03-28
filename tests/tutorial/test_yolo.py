from argparse import ArgumentParser
import numpy as np
from mblt_models.vision import YOLOv5m

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="/workspace/mblt-models/tests/rc/cr7.jpg"
    )
    args = parser.parse_args()
    image_path = args.image_path

    yolov5m = YOLOv5m(local_model="/workspace/mblt-models/tmp/yolov5m.mxq")
    yolov5m_pre = yolov5m.get_preprocess()
    yolov5m_post = yolov5m.get_postprocess(conf_thres=0.5, iou_thres=0.5)

    input = yolov5m_pre(image_path)
    output = yolov5m(input)
    result = yolov5m_post(output)
    print(result)
