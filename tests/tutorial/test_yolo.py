from argparse import ArgumentParser
from mblt_model_zoo.vision import YOLOv8m

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="/workspace/mblt-model-zoo/tests/rc/cr7.jpg"
    )
    args = parser.parse_args()
    image_path = args.image_path

    yolo = YOLOv8m()

    input_img = yolo.preprocess(image_path)
    output = yolo(input_img)
    result = yolo.postprocess(output, conf_thres=0.5, iou_thres=0.5)

    result.plot(
        source_path=image_path,
        save_path="/workspace/mblt-model-zoo/tests/tmp/cr7_yolov8m.jpg",
    )
