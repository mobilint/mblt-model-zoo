"""Evaluation script for COCO dataset."""

import logging
import math
import os
from time import time

import torch
from faster_coco_eval import COCO, COCOeval_faster
from tqdm import tqdm

from ..datasets import CustomCocodata, get_coco_inv, get_coco_loader
from ..postprocess.common import scale_boxes, scale_coords, scale_masks

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def to_string(counts: list[int]) -> str:
    """Converts the RLE object into a compact string representation.

    Each count is delta-encoded and variable-length encoded as a string.

    Args:
        counts (list[int]): List of RLE counts.

    Returns:
        str: Compact string representation of the RLE object.
    """
    result = []

    for i, x in enumerate(counts):
        x = int(x)

        # Apply delta encoding for all counts after the second entry
        if i > 2:
            x -= int(counts[i - 2])

        # Variable-length encode the value
        while True:
            c = x & 0x1F  # Take 5 bits
            x >>= 5

            # If the sign bit (0x10) is set, continue if x != -1;
            # otherwise, continue if x != 0
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20  # Set continuation bit
            c += 48  # Shift to ASCII
            result.append(chr(c))
            if not more:
                break

    return "".join(result)


def multi_encode(pixels: torch.Tensor) -> list[int]:
    """Convert multiple binary masks using Run-Length Encoding (RLE).

    Args:
        pixels (torch.Tensor): A 2D tensor where each row represents a flattened binary mask
            with shape [N, H*W].

    Returns:
        list[list[int]]: A list of RLE counts for each mask.
    """
    transitions = pixels[:, 1:] != pixels[:, :-1]
    row_idx, col_idx = torch.where(transitions)
    col_idx = col_idx + 1

    # Compute run lengths
    counts = []
    for i, pixel_row in enumerate(pixels):
        positions = col_idx[row_idx == i]
        if len(positions):
            count = torch.diff(positions).tolist()
            count.insert(0, positions[0].item())
            count.append(len(pixel_row) - positions[-1].item())
        else:
            count = [len(pixel_row)]

        # Ensure starting with background (0) count
        if pixel_row[0].item() == 1:
            count = [0, *count]
        counts.append(count)

    return counts


def nmsout2eval(nms_outs, img1_shape, img0_shapes):
    """Converts NMS output to COCO evaluation format.

    Args:
        nms_outs (List[np.array or torch.Tensor] or torch.Tensor): The output of the NMS
            operation of shape (n, 6), where n is the number of objects.
        img1_shape (tuple): Processed image shape (H, W).
        img0_shapes (list[tuple]): Original image shapes [(H, W), ...].

    Returns:
        tuple: A tuple containing:
            - labels (list[list]): The labels of the objects for each image.
            - boxes (list[list]): The bounding boxes (xywh) for each image.
            - scores (list[list]): The confidence scores for each image.
    """

    if not isinstance(nms_outs, list):
        nms_outs = [nms_outs]
    labels_list = []
    boxes_list = []
    scores_list = []
    for nms_out, img0_shape in zip(nms_outs, img0_shapes):
        boxes = nms_out[:, :4]
        scores = nms_out[:, 4]
        labels = nms_out[:, 5]
        boxes = scale_boxes(
            img1_shape, boxes, img0_shape
        )  # scale boxes to original image size
        boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]  # xyxy to xywh with corner xy

        boxes = boxes.tolist()
        scores = scores.tolist()
        labels = labels.tolist()
        labels = [get_coco_inv(int(l)) for l in labels]

        labels_list.append(labels)
        boxes_list.append(boxes)
        scores_list.append(scores)

    return labels_list, boxes_list, scores_list


def nmsout2eval_seg(nms_outs, img1_shape, img0_shapes):
    """Converts segmentation NMS output to COCO evaluation format.

    Args:
        nms_outs (list): The output of the NMS operation.
        img1_shape (tuple): Processed image shape (H, W).
        img0_shapes (list[tuple]): Original image shapes [(H, W), ...].

    Returns:
        tuple: A tuple containing:
            - labels (list[list]): The labels of the objects for each image.
            - boxes (list[list]): The bounding boxes (xywh) for each image.
            - scores (list[list]): The confidence scores for each image.
            - extra (list[list]): The encoded segmentation masks for each image.
    """
    if not isinstance(nms_outs[0], list):
        nms_outs = [nms_outs]

    det_results = []
    seg_results = []
    for nms_out in nms_outs:
        det_results.append(nms_out[0])
        seg_results.append(nms_out[1])

    labels_list, boxes_list, scores_list = nmsout2eval(
        det_results, img1_shape, img0_shapes
    )

    seg_results = [
        scale_masks(seg_result.to(torch.float32), (img0_shape[0], img0_shape[1]))
        for seg_result, img0_shape in zip(seg_results, img0_shapes)
    ]

    def mask_encode(seg_result):
        extra = []
        h, w = seg_result.shape[1:3]
        seg_result = (
            seg_result.permute(0, 2, 1)
            .contiguous()
            .view(seg_result.shape[0], h * w)
            .byte()
        )
        counts = multi_encode(seg_result)
        for c in counts:
            extra.append({"size": [h, w], "counts": to_string(c)})
        return extra

    extra_list = [mask_encode(seg_result) for seg_result in seg_results]
    return labels_list, boxes_list, scores_list, extra_list


def nmsout2eval_pose(nms_outs, img1_shape, img0_shapes):
    """Converts pose estimation NMS output to COCO evaluation format.

    Args:
        nms_outs (list): The output of the NMS operation.
        img1_shape (tuple): Processed image shape (H, W).
        img0_shapes (list[tuple]): Original image shapes [(H, W), ...].

    Returns:
        tuple: A tuple containing:
            - labels (list[list]): The labels of the objects for each image.
            - boxes (list[list]): The bounding boxes (xywh) for each image.
            - scores (list[list]): The confidence scores for each image.
            - keypoints (list[list]): The scaled keypoints for each image.
    """
    if not isinstance(nms_outs, list):
        nms_outs = [nms_outs]
    labels_list, boxes_list, scores_list = nmsout2eval(
        nms_outs, img1_shape, img0_shapes
    )
    extra = [
        scale_coords(img1_shape, nms_out[:, 6:].reshape(-1, 17, 3), img0_shape).reshape(
            -1, 51
        )
        for nms_out, img0_shape in zip(nms_outs, img0_shapes)
    ]
    return labels_list, boxes_list, scores_list, extra


def format_coco_results(task, nms_outs, input_shape, org_shape, idx, dataset_ids):
    """Format the results for COCO evaluation.
    Args:
        task (str): The task to evaluate.
        nms_outs (MBLT_PostProcessResult): The output of the postprocessing.
        input_shape (tuple): The shape of the input tensor.
        org_shape (tuple): The original shape of the image.
        idx (list): The indices of the images in the batch.
        dataset_ids (list): The list of image IDs in the dataset.
    Returns:
        list: The formatted results.
    """
    results = []
    if task == "object_detection":
        labels_list, boxes_list, scores_list = nmsout2eval(
            nms_outs.output, input_shape, org_shape
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
        labels_list, boxes_list, scores_list, extra_list = nmsout2eval_seg(
            nms_outs.output, input_shape, org_shape
        )
        for i, labels, boxes, scores, extra in zip(
            idx, labels_list, boxes_list, scores_list, extra_list
        ):
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
        labels_list, boxes_list, scores_list, extra_list = nmsout2eval_pose(
            nms_outs.output, input_shape, org_shape
        )
        for i, labels, boxes, scores, extra in zip(
            idx, labels_list, boxes_list, scores_list, extra_list
        ):
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
            f"Only object detection, instance segmentation, and "
            f"pose estimation are supported, but we got {task}"
        )
    return results


def eval_coco(model, data_path, batch_size, conf_thres, iou_thres):
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

    inference_time = 0
    infer_post_time = 0
    total_time = 0

    cum_num_data = 0

    for input_npu, org_shape, idx in pbar:
        cum_num_data += len(idx)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic

        nms_outs = model.postprocess(
            out_npu, conf_thres=conf_thres, iou_thres=iou_thres
        )
        infer_post_time += time() - tic
        results.extend(
            format_coco_results(
                model.post_cfg["task"],
                nms_outs,
                input_npu.shape[1:-1],
                org_shape,
                idx,
                dataset.ids,
            )
        )

        total_time += time() - tic
        pbar.set_postfix_str(f"NPU FPS: {cum_num_data / inference_time:.3f}")

    pbar.close()
    res = evaluate_predictions_on_coco(dataset.coco, results, model.post_cfg["task"])

    print("COCO evaluation completed")
    return res.stats[0].item()


def evaluate_predictions_on_coco(coco_gt, coco_results: dict, task: str):
    """Evaluates predictions using the COCO API.

    Args:
        coco_gt (COCO): Ground truth COCO object.
        coco_results (dict): Predictions in COCO format.
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
