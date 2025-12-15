import logging
import math
import os
from multiprocessing.pool import ThreadPool
from time import time

import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster, mask
from tqdm import tqdm

from ...utils.datasets.coco import get_coco_inv
from ...utils.postprocess.common import scale_boxes, scale_coords, scale_image
from ..dataloader import CustomCocodata, get_coco_loader

NUM_THREADS = min(16, max(1, os.cpu_count() - 1))


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def single_encode(x):
    """Encode predicted masks as RLE and append results to jdict."""
    rle = mask.encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def nmsout2eval(nms_out, img1_shape, img0_shape):
    """NMS output to evaluation format.

    Args:
        nms_out (np. array or torch.Tensor): The output of the NMS operation of shape (n, 6), where n is the number of objects
        img1_shape (tuple): processed image shape
        img0_shape (tuple): original image shape

    Returns:
        labels (list): The labels of the objects
        boxes (list): The bounding boxes of the objects
        scores (list): The confidence scores of the objects
    """
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

    return labels, boxes, scores


def nmsout2eval_seg(nms_out, img1_shape, img0_shape):
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
    det_result = nms_out[0]
    seg_result = nms_out[1].to(torch.uint8).cpu().numpy()

    labels, boxes, scores = nmsout2eval(det_result, img1_shape, img0_shape)
    masks = scale_image(seg_result.transpose(1, 2, 0), img0_shape)  # HWC
    with ThreadPool(NUM_THREADS) as pool:
        extra = pool.map(
            single_encode, np.transpose(masks, (2, 0, 1))
        )  # RLE encode. mask shape: (c, h, w)
    return labels, boxes, scores, extra


def nmsout2eval_pose(nms_out, img1_shape, img0_shape):
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
    labels, boxes, scores = nmsout2eval(nms_out, img1_shape, img0_shape)
    extra = scale_coords(img1_shape, nms_out[:, 6:].reshape(-1, 17, 3), img0_shape)
    return labels, boxes, scores, extra.reshape(-1, 51).tolist()


def eval_coco(model, data_path, batch_size):

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
