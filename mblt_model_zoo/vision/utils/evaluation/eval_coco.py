"""Evaluation script for COCO dataset."""

from __future__ import annotations

import logging
import math
import os
from time import time
from typing import TYPE_CHECKING, Any, NamedTuple

from faster_coco_eval import COCO, COCOeval_faster
from tqdm import tqdm

from ..._tasks import normalize_vision_task
from ..datasets import CustomCOCODataset, get_coco_loader

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine
    from ..results import Results

logger = logging.getLogger(__name__)


class COCOResult(NamedTuple):
    """COCO mAP metrics."""

    map5095: float
    map50: float

    @property
    def primary_score(self) -> float:
        """Return mAP50-95."""

        return self.map5095

    @property
    def secondary_score(self) -> float:
        """Return mAP50."""

        return self.map50


def format_coco_results(
    task: str,
    nms_outs: Results,
    input_shape: tuple[int, ...],
    org_shape: tuple[int, ...],
    ratio_pad: list[Any],
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
        labels_list, boxes_list, scores_list = postprocess.nmsout2eval(
            nms_outs.output,
            input_shape,
            org_shape,
            ratio_pad=ratio_pad,
        )
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
            nms_outs.output,
            input_shape,
            org_shape,
            ratio_pad=ratio_pad,
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
            nms_outs.output,
            input_shape,
            org_shape,
            ratio_pad=ratio_pad,
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


def eval_coco(
    model: MBLT_Engine,
    data_path: str,
    batch_size: int,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
) -> float:
    """Evaluate a model on COCO and return the legacy numeric mAP50-95 score."""

    return eval_coco_metrics(model, data_path, batch_size, conf_thres, iou_thres).primary_score


def eval_coco_metrics(
    model: MBLT_Engine,
    data_path: str,
    batch_size: int,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
) -> COCOResult:
    """Evaluate a model on COCO and return structured mAP metrics.

    Args:
        model (MBLT_Engine): The model engine to evaluate.
        data_path (str): Path to the COCO dataset.
        batch_size (int): Batch size for evaluation.
        conf_thres (float | None): Optional confidence threshold override.
        iou_thres (float | None): Optional IoU threshold override.

    Returns:
        Structured mAP50-95 primary and mAP50 secondary metrics.
    """
    task = normalize_vision_task(
        model.post_cfg["task"],
        supported=("object_detection", "instance_segmentation", "pose_estimation"),
    )
    if task in {"object_detection", "instance_segmentation"}:
        dataset = CustomCOCODataset(
            os.path.join(data_path, "val2017"),
            os.path.join(data_path, "instances_val2017.json"),
        )
    else:
        dataset = CustomCOCODataset(
            os.path.join(data_path, "val2017"),
            os.path.join(data_path, "person_keypoints_val2017.json"),
            min_keypoints=0,
        )

    dataloader = get_coco_loader(dataset, batch_size, model.preprocess_with_metadata)
    model.set_postprocess_thresholds(conf_thres=conf_thres, iou_thres=iou_thres)

    results = []
    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter, desc="Evaluating COCO")

    inference_time = 0.0
    cum_num_data = 0

    for input_npu, org_shape, ratio_pad, idx in pbar:
        cum_num_data += len(idx)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic

        nms_outs = model.postprocess(out_npu, multi_label=True)
        results.extend(
            format_coco_results(
                task,
                nms_outs,
                input_npu.shape[1:-1],
                org_shape,
                ratio_pad,
                idx,
                dataset.ids,
                model.postprocessor,
            )
        )

        pbar.set_postfix_str(f"NPU FPS: {cum_num_data / inference_time:.3f}")

    pbar.close()
    res = evaluate_predictions_on_coco(dataset.coco, results, task, img_ids=dataset.ids)

    print("COCO evaluation completed")
    return COCOResult(map5095=float(res.stats[0].item()), map50=float(res.stats[1].item()))


def evaluate_predictions_on_coco(
    coco_gt: COCO,
    coco_results: list[dict[str, Any]],
    task: str,
    img_ids: list[int] | None = None,
) -> COCOeval_faster:
    """Evaluates predictions using the COCO API.

    Args:
        coco_gt (COCO): Ground truth COCO object.
        coco_results (list): Predictions in COCO format.
        task (str): Task type ('object_detection', 'instance_segmentation', or 'pose_estimation').
        img_ids: Optional image IDs to include in evaluation.

    Returns:
        COCOeval_faster: The COCO evaluation object containing results.
    """
    normalized_task = normalize_vision_task(
        task,
        supported=("object_detection", "instance_segmentation", "pose_estimation"),
    )

    if coco_results:
        coco_dt = coco_gt.loadRes(coco_results)
    else:
        coco_dt = COCO()

    if normalized_task == "object_detection":
        coco_eval = COCOeval_faster(coco_gt, coco_dt, "bbox", print_function=logger.info)
    elif normalized_task == "instance_segmentation":
        coco_eval = COCOeval_faster(coco_gt, coco_dt, "segm", print_function=logger.info)
    elif normalized_task == "pose_estimation":
        coco_eval = COCOeval_faster(coco_gt, coco_dt, "keypoints", print_function=logger.info)
    else:
        raise RuntimeError(f"Unexpected validated COCO task: {normalized_task}")

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
