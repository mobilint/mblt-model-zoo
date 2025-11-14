import json
import logging
import math
import tempfile
from time import time

from faster_coco_eval import COCO, COCOeval_faster
from tqdm import tqdm

from ...wrapper import MBLT_Engine
from ..datasets import load_dataloader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def eval_coco(
    model: MBLT_Engine,
    coco_eval_cfg: dict,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    batch_size: int = 1,
):
    """Evaluate the model on the coco dataset

    Args:
        model (Model Class): Model class object
        helper (Helper Class): Helper class object with pre_process and post_process methods
        coco_eval_cfg (dict): Configuration dictionary for the evaluation

    Returns:
        mAP, inference_time, fps: Mean Average Precision, Inference time, and Frames per second
    """
    if model._postprocess.task.lower() in [
        "object_detection",
        "instance_segmentation",
    ]:
        dataset, dataloader = load_dataloader(
            dataset_name="coco",
            dataset_path=coco_eval_cfg["dataset_path"],
            preprocess=model.preprocess,
            annotation_path=coco_eval_cfg["annotation_path"],
            batch_size=batch_size,
        )
    elif model._postprocess.task.lower() == "pose_estimation":
        dataset, dataloader = load_dataloader(
            dataset_name="coco",
            dataset_path=coco_eval_cfg["dataset_path"],
            preprocess=model.preprocess,
            annotation_path=coco_eval_cfg["keypoint_annotation_path"],
            batch_size=batch_size,
        )
    else:
        raise NotImplementedError(f"Task {model._postprocess.task} is not supported")

    results = []
    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter)

    inference_time = 0
    infer_post_time = 0
    total_time = 0
    cum_num_data = 0

    for input_npu, org_shape, idx in pbar:
        cum_num_data += len(idx)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic

        nms_outs = model.postprocess(out_npu)
        infer_post_time += time() - tic

        if model._postprocess.task.lower() == "object_detection":
            labels_list, boxes_list, scores_list = model._postprocess.nmsout2eval(
                nms_outs, input_npu.shape[-2:], org_shape
            )
            for i, labels, boxes, scores in zip(
                idx, labels_list, boxes_list, scores_list
            ):
                results.extend(
                    [
                        {
                            "image_id": dataset.ids[i],
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
        elif model._postprocess.task.lower() == "instance_segmentation":
            (
                labels_list,
                boxes_list,
                scores_list,
                extra_list,
            ) = model._postprocess.nmsout2eval(
                nms_outs, input_npu.shape[-2:], org_shape
            )
            for i, labels, boxes, scores, extra in zip(
                idx, labels_list, boxes_list, scores_list, extra_list
            ):
                results.extend(
                    [
                        {
                            "image_id": dataset.ids[i],
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                            "segmentation": extra[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
        elif model._postprocess.task.lower() == "pose_estimation":
            (
                labels_list,
                boxes_list,
                scores_list,
                extra_list,
            ) = model._postprocess.nmsout2eval(
                nms_outs, input_npu.shape[-2:], org_shape
            )
            for i, labels, boxes, scores, extra in zip(
                idx, labels_list, boxes_list, scores_list, extra_list
            ):
                results.extend(
                    [
                        {
                            "image_id": dataset.ids[i],
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                            "keypoints": extra[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
        else:
            raise NotImplementedError(
                f"Only object detection, instance segmentation, and pose estimation are supported, but we got {model._postprocess.task}"
            )
        total_time += time() - tic
        pbar.set_postfix_str(
            f"FPS: {cum_num_data / inference_time:.2f}, Extended FPS: {cum_num_data/ infer_post_time:.2f}, Total FPS: {cum_num_data / total_time:.2f}"
        )

    with tempfile.NamedTemporaryFile() as f:
        file_path = f.name
        res = evaluate_predictions_on_coco(
            dataset.coco, results, file_path, model._postprocess.task
        )

    return (
        res.stats[0].item(),
        inference_time,
        num_data / inference_time,
        infer_post_time,
        num_data / infer_post_time,
        total_time,
        num_data / total_time,
    )


def evaluate_predictions_on_coco(coco_gt, coco_results, json_result_file, task):
    """Evaluate the predictions on the coco dataset

    Args:
        coco_gt: Ground truth coco object
        coco_results: Results to evaluate
        json_result_file: File to save the results
        task (str): Task to evaluate

    Returns:
        _type_: _description_
    """
    assert task in [
        "object_detection",
        "instance_segmentation",
        "pose_estimation",
    ], f"task should be included in [detection, seg, pose] but we got {task}"
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)
    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
    if task == "object_detection":
        coco_eval = COCOeval_faster(
            coco_gt, coco_dt, "bbox", print_function=logger.info
        )
    elif task == "instance_segmentation":
        coco_eval = COCOeval_faster(
            coco_gt, coco_dt, "segm", print_function=logger.info
        )
    elif task == "pose_estimation":
        coco_eval = COCOeval_faster(
            coco_gt, coco_dt, "keypoints", print_function=logger.info
        )
    else:
        raise NotImplementedError(f"Task {task} is not supported")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
