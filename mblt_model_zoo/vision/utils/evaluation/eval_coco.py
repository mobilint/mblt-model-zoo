"""Evaluation script for COCO dataset."""

from __future__ import annotations

import logging
import math
import os
from time import time
from typing import TYPE_CHECKING, Any

from faster_coco_eval import COCO, COCOeval_faster
from tqdm import tqdm

from ..datasets import CustomCocodata, get_coco_loader

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine
    from ..results import Results

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def format_coco_results(
    task: str,
    nms_outs: Results,
    input_shape: tuple[int, ...],
    org_shape: tuple[int, ...],
    idx: list[int],
    dataset_ids: list[int],
    postprocess: Any,
) -> list[dict[str, Any]]:
    """Format the results for COCO evaluation.
    Args:
        task (str): The task to evaluate.
        nms_outs (Results): The output of the postprocessing.
        input_shape (tuple): The shape of the input tensor.
        org_shape (tuple): The original shape of the image.
        idx (list): The indices of the images in the batch.
        dataset_ids (list): The list of image IDs in the dataset.
        postprocess: The postprocessing instance.
    Returns:
        list: The formatted results.
    """
    results = []
    if task == "object_detection":
        labels_list, boxes_list, scores_list = postprocess.nmsout2eval(nms_outs.output, input_shape, org_shape)
        for i, labels, boxes, scores in zip(idx, labels_list, boxes_list, scores_list):
            results.extend(
                [
                    {
                        "image_id": dataset_ids[i],
                        "category_id": label,
                        "bbox": box,
                        "score": score,
                    }
                    for box, score, label in zip(boxes, scores, labels)
                ]
            )
    elif task == "instance_segmentation":
        labels_list, boxes_list, scores_list, extra_list = postprocess.nmsout2eval(
            nms_outs.output, input_shape, org_shape
        )
        for i, labels, boxes, scores, extra in zip(idx, labels_list, boxes_list, scores_list, extra_list):
            results.extend(
                [
                    {
                        "image_id": dataset_ids[i],
                        "category_id": label,
                        "bbox": box,
                        "score": score,
                        "segmentation": extra,
                    }
                    for box, score, label, extra in zip(boxes, scores, labels, extra)
                ]
            )
    elif task == "pose_estimation":
        labels_list, boxes_list, scores_list, extra_list = postprocess.nmsout2eval(
            nms_outs.output, input_shape, org_shape
        )
        for i, labels, boxes, scores, extra in zip(idx, labels_list, boxes_list, scores_list, extra_list):
            results.extend(
                [
                    {
                        "image_id": dataset_ids[i],
                        "category_id": label,
                        "bbox": box,
                        "score": score,
                        "keypoints": extra,
                    }
                    for box, score, label, extra in zip(boxes, scores, labels, extra)
                ]
            )
    else:
        raise NotImplementedError(
            f"Only object detection, instance segmentation, and pose estimation are supported, but we got {task}"
        )
    return results


def eval_coco(model: MBLT_Engine, data_path: str, batch_size: int, conf_thres: float, iou_thres: float) -> float:
    """Evaluates a model on the COCO dataset.

    Args:
        model (MBLT_Engine): The model engine to evaluate.
        data_path (str): Path to the COCO dataset.
        batch_size (int): Batch size for evaluation.
        conf_thres (float): Confidence threshold for detection.
        iou_thres (float): IoU threshold for NMS.

    Returns:
        float: The mAP score (average precision at IoU=0.50:0.95).
    """
    if model.post_cfg["task"] in ["object_detection", "instance_segmentation"]:
        dataset = CustomCocodata(
            os.path.join(data_path, "val2017"),
            os.path.join(data_path, "instances_val2017.json"),
        )
    elif model.post_cfg["task"] == "pose_estimation":
        dataset = CustomCocodata(
            os.path.join(data_path, "val2017"),
            os.path.join(data_path, "person_keypoints_val2017.json"),
        )
    else:
        raise NotImplementedError(f"Task {model.post_cfg['task']} is not supported")

    dataloader = get_coco_loader(dataset, batch_size, model.preprocess)

    results = []
    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter, desc="Evaluating COCO")

    inference_time = 0.0
    infer_post_time = 0.0
    total_time = 0.0

    cum_num_data = 0

    for input_npu, org_shape, idx in pbar:
        cum_num_data += len(idx)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic

        nms_outs = model.postprocess(out_npu, conf_thres=conf_thres, iou_thres=iou_thres)
        infer_post_time += time() - tic
        results.extend(
            format_coco_results(
                model.post_cfg["task"],
                nms_outs,
                input_npu.shape[1:-1],
                org_shape,
                idx,
                dataset.ids,
                model.postprocessor,
            )
        )

        total_time += time() - tic
        pbar.set_postfix_str(f"NPU FPS: {cum_num_data / inference_time:.3f}")

    pbar.close()
    res = evaluate_predictions_on_coco(dataset.coco, results, model.post_cfg["task"])

    print("COCO evaluation completed")
    return float(res.stats[0].item())


def evaluate_predictions_on_coco(coco_gt: COCO, coco_results: list[dict[str, Any]], task: str) -> COCOeval_faster:
    """Evaluates predictions using the COCO API.

    Args:
        coco_gt (COCO): Ground truth COCO object.
        coco_results (list): Predictions in COCO format.
        task (str): Task type ('object_detection', 'instance_segmentation', or 'pose_estimation').

    Returns:
        COCOeval_faster: The COCO evaluation object containing results.
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
        coco_eval = COCOeval_faster(coco_gt, coco_dt, "bbox", print_function=logger.info)
    elif task.lower() == "instance_segmentation":
        coco_eval = COCOeval_faster(coco_gt, coco_dt, "segm", print_function=logger.info)
    elif task.lower() == "pose_estimation":
        coco_eval = COCOeval_faster(coco_gt, coco_dt, "keypoints", print_function=logger.info)
    else:
        raise NotImplementedError(f"Task {task} is not supported")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
