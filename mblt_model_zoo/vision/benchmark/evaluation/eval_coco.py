import logging
import math
import os
from time import time

import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster, mask
from tqdm import tqdm

from ...utils.datasets.coco import get_coco_inv
from ...utils.postprocess.common import scale_boxes, scale_coords, scale_image
from ..dataloader import CustomCocodata, get_coco_loader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def single_encode(x):
    """Encode predicted masks as RLE and append results to jdict."""
    rle = mask.encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def nmsout2eval(nms_outs, img1_shape, img0_shapes):
    """NMS output to evaluation format.

    Args:
        nms_outs (List[np.array or torch.Tensor] or torch.Tensor): The output of the NMS operation of shape (n, 6), where n is the number of objects
        img1_shape (tuple): processed image shape
        img0_shape (tuple): original image shape

    Returns:
        labels (list): The labels of the objects
        boxes (list): The bounding boxes of the objects
        scores (list): The confidence scores of the objects
    """

    if not isinstance(nms_outs, list):
        nms_outs = [nms_outs]
    lebels_list = []
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

        lebels_list.append(labels)
        boxes_list.append(boxes)
        scores_list.append(scores)

    return lebels_list, boxes_list, scores_list


def nmsout2eval_seg(nms_outs, img1_shape, img0_shapes):
    """NMS output to evaluation format.

    Args:
        nms_out (np.array or torch.Tensor): The output of the NMS operation of shape (n, 6), where n is the number of objects
        img1_shape (tuple): processed image shape
        img0_shape (tuple): original image shape

    Returns:
        labels (list): The labels of the objects
        boxes (list): The bounding boxes of the objects
        scores (list): The confidence scores of the objects
        extra (list): The segmentation masks of the objects
    """
    if not isinstance(nms_outs[0], list):
        nms_outs = [nms_outs]

    det_results = []
    seg_results = []
    for nms_out in nms_outs:
        det_results.append(nms_out[0])
        seg_results.append(nms_out[1].to(torch.uint8).cpu().numpy())

    labels_list, boxes_list, scores_list = nmsout2eval(
        det_results, img1_shape, img0_shapes
    )

    masks = [
        scale_image(seg_result.transpose(1, 2, 0), img0_shape)
        for seg_result, img0_shape in zip(seg_results, img0_shapes)
    ]  # HWC
    extra_list = [
        [
            single_encode(mask_channel)
            for mask_channel in np.transpose(mask, (2, 0, 1))  # hwc -> chw
        ]  # RLE encode. mask_channel shape: (h, w)
        for mask in masks
    ]
    return labels_list, boxes_list, scores_list, extra_list


def nmsout2eval_pose(nms_outs, img1_shape, img0_shapes):
    """NMS output to evaluation format.

    Args:
        nms_out (np.array or torch.Tensor): The output of the NMS operation of shape (n, 6+51), where n is the number of objects
        img1_shape (tuple): processed image shape
        img0_shape (tuple): original image shape

    Returns:
        labels (list): The labels of the objects
        boxes (list): The bounding boxes of the objects
        scores (list): The confidence scores of the objects
        keypoints (list): The keypoints of the objects
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


def eval_coco(model, data_path, batch_size, conf_thres, iou_thres):
    """
    Evaluate a model on the COCO dataset.

    Args:
        model: The model to evaluate.
        data_path (str): Path to the COCO data.
        batch_size (int): Batch size for evaluation.
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
        if model.post_cfg["task"] == "object_detection":
            labels_list, boxes_list, scores_list = nmsout2eval(
                nms_outs.output, input_npu.shape[1:-1], org_shape
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
        elif model.post_cfg["task"] == "instance_segmentation":
            labels_list, boxes_list, scores_list, extra_list = nmsout2eval_seg(
                nms_outs.output, input_npu.shape[1:-1], org_shape
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
        elif model.post_cfg["task"] == "pose_estimation":
            labels_list, boxes_list, scores_list, extra_list = nmsout2eval_pose(
                nms_outs.output, input_npu.shape[1:-1], org_shape
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
                f"Only object detection, instance segmentation, and pose estimation are supported, but we got {model.post_cfg['task']}"
            )

        total_time += time() - tic
        pbar.set_postfix_str(f"NPU FPS: {cum_num_data / inference_time:.3f}")

    pbar.close()
    res = evaluate_predictions_on_coco(dataset.coco, results, model.post_cfg["task"])

    print("COCO evaluation completed")
    return res.stats[0].item()


def evaluate_predictions_on_coco(coco_gt, coco_results: dict, task: str):
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
