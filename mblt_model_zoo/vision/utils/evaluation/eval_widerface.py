"""Evaluation script for WiderFace face detection."""

from __future__ import annotations

import math
import os
from time import time
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from ..datasets import CustomWiderface, get_widerface_loader
from ..postprocess.base import YOLOPostBase

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine


class WiderFaceResult(NamedTuple):
    """WiderFace AP metrics."""

    easy_ap: float
    medium_ap: float
    hard_ap: float

    @property
    def mean_ap(self) -> float:
        """Return the mean AP across Easy, Medium, and Hard."""

        return (self.easy_ap + self.medium_ap + self.hard_ap) / 3.0


def _empty_prediction() -> np.ndarray:
    """Return an empty WiderFace prediction array."""

    return np.zeros((0, 5), dtype=np.float32)


def _initialize_predictions(dataset: CustomWiderface) -> dict[str, dict[str, np.ndarray]]:
    """Initialize empty predictions for every WiderFace sample."""

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for _, event_name, file_name in dataset.samples:
        predictions.setdefault(event_name, {})[os.path.splitext(file_name)[0]] = _empty_prediction()
    return predictions


def _boxes_scores_to_prediction(boxes: list[list[float]], scores: list[float]) -> np.ndarray:
    """Convert xywh boxes and scores to a WiderFace prediction array."""

    if not boxes:
        return _empty_prediction()
    prediction = np.zeros((len(boxes), 5), dtype=np.float32)
    for index, (box, score) in enumerate(zip(boxes, scores)):
        prediction[index, :4] = np.asarray(box, dtype=np.float32)
        prediction[index, 4] = float(score)
    return prediction


def eval_widerface(
    model: MBLT_Engine,
    data_path: str,
    batch_size: int,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
) -> WiderFaceResult:
    """Evaluate a face-detection model on WiderFace validation data.

    Args:
        model: The face-detection engine to evaluate.
        data_path: Organized WiderFace dataset root.
        batch_size: Validation batch size.
        conf_thres: Optional confidence threshold override.
        iou_thres: Optional IoU threshold override.

    Returns:
        WiderFace Easy, Medium, and Hard AP metrics.
    """

    if model.post_cfg["task"] != "face_detection":
        raise NotImplementedError(f"Task {model.post_cfg['task']} is not supported for WiderFace evaluation.")

    dataset = CustomWiderface(os.path.join(data_path, "images"))
    dataloader = get_widerface_loader(dataset, batch_size, model.preprocess_with_metadata)
    model.set_postprocess_thresholds(conf_thres=conf_thres, iou_thres=iou_thres)

    predictions = _initialize_predictions(dataset)
    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter, desc="Evaluating WiderFace")
    inference_time = 0.0
    cum_num_data = 0

    for input_npu, org_shape, ratio_pad, target_classes, fnames in pbar:
        cum_num_data += len(fnames)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic
        nms_outs = model.postprocess(out_npu)
        input_shape = (int(input_npu.shape[1]), int(input_npu.shape[2]))
        img0_shapes = [(int(shape[0]), int(shape[1])) for shape in org_shape.tolist()]
        postprocessor = cast(YOLOPostBase, model.postprocessor)
        _, boxes_list, scores_list = postprocessor.nmsout2eval(
            nms_outs.output,
            input_shape,
            img0_shapes,
            ratio_pad=ratio_pad,
        )

        for event_name, file_name, boxes, scores in zip(target_classes, fnames, boxes_list, scores_list):
            predictions[event_name][os.path.splitext(file_name)[0]] = _boxes_scores_to_prediction(boxes, scores)

        pbar.set_postfix_str(f"NPU FPS: {cum_num_data / inference_time:.3f}")

    pbar.close()
    aps = evaluation(norm_score(predictions), data_path)
    print("WiderFace evaluation completed")
    return WiderFaceResult(*aps)


def bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU overlaps between boxes and query boxes."""

    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)

    boxes = boxes[:, None, :]
    query_boxes = query_boxes[None, :, :]

    iw = np.minimum(boxes[..., 2], query_boxes[..., 2]) - np.maximum(boxes[..., 0], query_boxes[..., 0]) + 1
    ih = np.minimum(boxes[..., 3], query_boxes[..., 3]) - np.maximum(boxes[..., 1], query_boxes[..., 1]) + 1
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    inter = iw * ih

    box_area = (boxes[..., 2] - boxes[..., 0] + 1) * (boxes[..., 3] - boxes[..., 1] + 1)
    query_area = (query_boxes[..., 2] - query_boxes[..., 0] + 1) * (query_boxes[..., 3] - query_boxes[..., 1] + 1)
    union = box_area + query_area - inter

    return inter / union


def get_gt_boxes(gt_dir: str) -> tuple[Any, ...]:
    """Load WiderFace evaluation `.mat` files from the organized dataset."""

    gt_mat = loadmat(os.path.join(gt_dir, "wider_face_val.mat"))
    hard_mat = loadmat(os.path.join(gt_dir, "wider_hard_val.mat"))
    medium_mat = loadmat(os.path.join(gt_dir, "wider_medium_val.mat"))
    easy_mat = loadmat(os.path.join(gt_dir, "wider_easy_val.mat"))

    facebox_list = gt_mat["face_bbx_list"]
    event_list = gt_mat["event_list"]
    file_list = gt_mat["file_list"]
    hard_gt_list = hard_mat["gt_list"]
    medium_gt_list = medium_mat["gt_list"]
    easy_gt_list = easy_mat["gt_list"]

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def norm_score(pred: dict[str, Any]) -> dict[str, Any]:
    """Normalize WiderFace prediction scores to ``[0, 1]``."""

    max_score = -1e9
    min_score = 1e9
    found = False

    for _, event_predictions in pred.items():
        for _, image_predictions in event_predictions.items():
            if len(image_predictions) == 0:
                continue
            found = True
            _min = float(np.min(image_predictions[:, -1]))
            _max = float(np.max(image_predictions[:, -1]))
            if _max > max_score:
                max_score = _max
            if _min < min_score:
                min_score = _min

    if not found:
        return pred

    diff = max_score - min_score
    if diff <= 0:
        return pred

    for _, event_predictions in pred.items():
        for _, image_predictions in event_predictions.items():
            if len(image_predictions) == 0:
                continue
            image_predictions[:, -1] = (image_predictions[:, -1] - min_score) / diff

    return pred


def image_eval(
    pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray, iou_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one image worth of WiderFace predictions."""

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for prediction_index in range(_pred.shape[0]):
        gt_overlap = overlaps[prediction_index]
        max_overlap = np.max(gt_overlap)
        max_idx = np.argmax(gt_overlap)
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[prediction_index] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[prediction_index] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(
    thresh_num: int, pred_info: np.ndarray, proposal_list: np.ndarray, pred_recall: np.ndarray
) -> np.ndarray:
    """Compute precision and recall contributions for one image."""

    pr_info = np.zeros((thresh_num, 2), dtype=np.float32)
    for threshold_index in range(thresh_num):
        thresh = 1 - (threshold_index + 1) / thresh_num
        recall_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(recall_index) == 0:
            pr_info[threshold_index, 0] = 0
            pr_info[threshold_index, 1] = 0
        else:
            last_index = recall_index[-1]
            proposal_index = np.where(proposal_list[: last_index + 1] == 1)[0]
            pr_info[threshold_index, 0] = len(proposal_index)
            pr_info[threshold_index, 1] = pred_recall[last_index]
    return pr_info


def dataset_pr_info(thresh_num: int, pr_curve: np.ndarray, count_face: int) -> np.ndarray:
    """Normalize a WiderFace precision-recall accumulator."""

    _pr_curve = np.zeros((thresh_num, 2), dtype=np.float32)
    for threshold_index in range(thresh_num):
        proposals = pr_curve[threshold_index, 0]
        matched = pr_curve[threshold_index, 1]
        _pr_curve[threshold_index, 0] = matched / proposals if proposals > 0 else 0.0
        _pr_curve[threshold_index, 1] = matched / count_face if count_face > 0 else 0.0
    return _pr_curve


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Compute VOC-style average precision."""

    mrec = np.concatenate((np.array([0.0]), rec, np.array([1.0])))
    mpre = np.concatenate((np.array([0.0]), prec, np.array([0.0])))

    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = np.maximum(mpre[index - 1], mpre[index])

    recall_change_index = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[recall_change_index + 1] - mrec[recall_change_index]) * mpre[recall_change_index + 1])
    return float(ap)


def evaluation(pred: dict[str, Any], gt_path: str, iou_thresh: float = 0.5) -> list[float]:
    """Evaluate WiderFace predictions against Easy, Medium, and Hard settings."""

    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ["easy", "medium", "hard"]
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2), dtype=np.float32)
        pbar = tqdm(range(event_num))
        for event_index in pbar:
            pbar.set_description(f"Processing {settings[setting_id]}")
            event_name = str(event_list[event_index][0][0])
            img_list = file_list[event_index][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[event_index][0]
            gt_bbx_list = facebox_list[event_index][0]
            for image_index, img_info in enumerate(img_list):
                pred_info = pred_list[str(img_info[0][0])]
                gt_boxes = np.array(gt_bbx_list[image_index][0], dtype=np.float32)
                keep_index = np.array(sub_gt_list[image_index][0], dtype=np.int64)
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)
                pr_curve += img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
        pbar.close()
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        aps.append(voc_ap(pr_curve[:, 1], pr_curve[:, 0]))

    print("==================== Results ====================")
    print(f"Easy   Val AP: {aps[0]}")
    print(f"Medium Val AP: {aps[1]}")
    print(f"Hard   Val AP: {aps[2]}")
    print("=================================================")
    return aps
