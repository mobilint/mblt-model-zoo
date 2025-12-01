import json
import logging
import math
from time import time

from faster_coco_eval import COCO, COCOeval_faster
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def eval_coco(model):
    pass


def evaluate_predictions_on_coco(coco_gt, coco_results, task):
    """Evaluate the predictions on the coco dataset

    Args:
        coco_gt: Ground truth coco object
        coco_results: Results to evaluate
        task (str): Task to evaluate

    Returns:
        _type_: _description_
    """
    assert task.lower() in [
        "object_detection",
        "instance_segmentation",
        "pose_estimation",
    ], f"task should be included in [detection, seg, pose] but we got {task.lower()}"

    if coco_results:
        coco_dt = coco_gt.loadRes(coco_results)
    else:
        coco_dt = COCO()

    if task.lower() == "object_detection":
        coco_eval = COCOeval_faster(
            coco_gt, coco_dt, "bbox", print_function=logger.info
        )
    elif task.lower() == "instance_segmentation":
        coco_eval = COCOeval_faster(
            coco_gt, coco_dt, "segm", print_function=logger.info
        )
    elif task.lower() == "pose_estimation":
        coco_eval = COCOeval_faster(
            coco_gt, coco_dt, "keypoints", print_function=logger.info
        )
    else:
        raise NotImplementedError(f"Task {task} is not supported")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
