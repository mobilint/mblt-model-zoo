from argparse import ArgumentParser
from mblt_model_zoo.vision import YOLOv5m, YOLOv5mSeg, YOLOv8m, YOLOv8mSeg, YOLOv8mPose
from mblt_model_zoo.vision.utils.results import Results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="/workspace/mblt-model-zoo/tests/rc/cr7.jpg"
    )
    args = parser.parse_args()
    image_path = args.image_path

    yolo = YOLOv8mSeg(local_model="/workspace/mblt-model-zoo/tmp/yolov8m-seg.mxq")
    yolo_pre = yolo.get_preprocess()
    yolo_post = yolo.get_postprocess(conf_thres=0.5, iou_thres=0.5)

    input = yolo_pre(image_path)
    output = yolo(input)
    result = Results.from_engine(yolo, yolo_post(output))

    result.plot(
        source_path=image_path,
        save_path="/workspace/mblt-model-zoo/tests/tmp/cr7_yolov8m_seg.jpg",
    )
