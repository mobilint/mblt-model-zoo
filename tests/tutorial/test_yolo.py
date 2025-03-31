from argparse import ArgumentParser
from mblt_model_zoo.vision import YOLOv5m, YOLOv5mSeg
from mblt_model_zoo.vision.utils.results import Results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="/workspace/mblt-model-zoo/tests/rc/cr7.jpg"
    )
    args = parser.parse_args()
    image_path = args.image_path

    yolo = YOLOv5m(local_model="/workspace/mblt-model-zoo/tmp/yolov5m.mxq")
    yolo_pre = yolo.get_preprocess()
    yolo_post = yolo.get_postprocess(conf_thres=0.5, iou_thres=0.5)

    input = yolo_pre(image_path)
    output = yolo(input)
    result = yolo_post(output)
    print(result)
