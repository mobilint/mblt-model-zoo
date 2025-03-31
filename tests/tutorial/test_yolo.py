from argparse import ArgumentParser
import numpy as np
from mblt_model_zoo.vision import YOLOv5mSeg

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="/workspace/mblt-model-zoo/tests/rc/cr7.jpg"
    )
    args = parser.parse_args()
    image_path = args.image_path

    yolov5m = YOLOv5mSeg(local_model="/workspace/mblt-model-zoo/tmp/yolov5m-seg.mxq")
    yolov5m_pre = yolov5m.get_preprocess()
    yolov5m_post = yolov5m.get_postprocess(conf_thres=0.5, iou_thres=0.5)

    input = yolov5m_pre(image_path)
    output = yolov5m(input)
    result = yolov5m_post(output)
    print(result)
