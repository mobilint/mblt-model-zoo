from argparse import ArgumentParser
from mblt_model_zoo.vision import YOLOv5m, YOLOv5mSeg, YOLOv8n, YOLOv8nSeg, YOLOv8nPose
import time


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="/workspace/mblt-model-zoo/tests/rc/cr7.jpg"
    )
    args = parser.parse_args()
    image_path = args.image_path

    yolo = YOLOv8nPose(local_model="/workspace/mblt-model-zoo/tmp/yolov8n-pose.mxq")

    input_img = yolo.preprocess(image_path)
    output = yolo(input_img)
    result = yolo.postprocess(output, conf_thres=0.5, iou_thres=0.5)

    result.plot(
        source_path=image_path,
        save_path="/workspace/mblt-model-zoo/tests/tmp/cr7_yolov8n_pose.jpg",
    )

    time.sleep(5)
    yolo.gpu()
    input_img = yolo.preprocess(image_path)
    output = yolo(input_img)
    result = yolo.postprocess(output, conf_thres=0.5, iou_thres=0.5)

    result.plot(
        source_path=image_path,
        save_path="/workspace/mblt-model-zoo/tests/tmp/cr7_yolov8n_pose_gpu.jpg",
    )

    time.sleep(5)
    yolo.cpu()
    input_img = yolo.preprocess(image_path)
    output = yolo(input_img)
    result = yolo.postprocess(output, conf_thres=0.5, iou_thres=0.5)

    result.plot(
        source_path=image_path,
        save_path="/workspace/mblt-model-zoo/tests/tmp/cr7_yolov8n_pose_cpu.jpg",
    )
