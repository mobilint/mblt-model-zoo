import math
import os
from time import time

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def eval_widerface(model):
    """
    Placeholder for WiderFace evaluation.

    Args:
        model: The model to evaluate.
    """
    pass


def bbox_overlaps(boxes, query_boxes):
    """
    Compute IoU (Intersection over Union) between two sets of boxes.

    Parameters
    ----------
    boxes : (N, 4) array
        Each box is [x1, y1, x2, y2].
    query_boxes : (K, 4) array
        Each box is [x1, y1, x2, y2].

    Returns
    -------
    overlaps : (N, K) array
        IoU values between each pair of boxes.
    """

    # Ensure they're float arrays
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)

    boxes = boxes[:, None, :]  # (N, 1, 4)
    query_boxes = query_boxes[None, :, :]  # (1, K, 4)

    # Intersection
    iw = (
        np.minimum(boxes[..., 2], query_boxes[..., 2])
        - np.maximum(boxes[..., 0], query_boxes[..., 0])
        + 1
    )
    ih = (
        np.minimum(boxes[..., 3], query_boxes[..., 3])
        - np.maximum(boxes[..., 1], query_boxes[..., 1])
        + 1
    )
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    inter = iw * ih  # (N, K)

    # Areas
    box_area = (boxes[..., 2] - boxes[..., 0] + 1) * (boxes[..., 3] - boxes[..., 1] + 1)
    query_area = (query_boxes[..., 2] - query_boxes[..., 0] + 1) * (
        query_boxes[..., 3] - query_boxes[..., 1] + 1
    )

    # Union
    union = box_area + query_area - inter

    # IoU
    overlaps = inter / union

    # Return numpy array directly
    return overlaps


def get_gt_boxes(gt_dir):
    """gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

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

    return (
        facebox_list,
        event_list,
        file_list,
        hard_gt_list,
        medium_gt_list,
        easy_gt_list,
    )


def norm_score(pred):
    """norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score) / diff

    return pred


def image_eval(pred, gt, ignore, iou_thresh):
    """single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

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

    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap = np.max(gt_overlap)
        max_idx = np.argmax(gt_overlap)
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    """
    Compute precision and recall information for a single image.

    Args:
        thresh_num (int): Number of thresholds to use.
        pred_info (np.ndarray): Predicted bounding boxes and scores.
        proposal_list (np.ndarray): Indicator of whether each prediction is a proposal.
        pred_recall (np.ndarray): Cumulative recall for each prediction.

    Returns:
        np.ndarray: Array containing precision and recall info for each threshold.
    """
    pr_info = np.zeros((thresh_num, 2), dtype=np.float32)
    for t in range(thresh_num):
        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[: r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    """
    Compute precision and recall information for the entire dataset.

    Args:
        thresh_num (int): Number of thresholds used.
        pr_curve (np.ndarray): Accumulated precision-recall curve data.
        count_face (int): Total number of ground truth faces in the dataset.

    Returns:
        np.ndarray: Normalized precision-recall curve.
    """
    _pr_curve = np.zeros((thresh_num, 2), dtype=np.float32)
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):
    """
    Compute VOC Average Precision (AP).

    Args:
        rec (np.ndarray): Recall values.
        prec (np.ndarray): Precision values.

    Returns:
        float: Computed Average Precision.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate((np.array([0.0]), rec, np.array([1.0])))
    mpre = np.concatenate((np.array([0.0]), prec, np.array([0.0])))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, iou_thresh=0.5):
    """
    Evaluate predictions against WiderFace ground truth.

    Args:
        pred (dict): Dictionary of predictions for each event and image.
        gt_path (str): Path to the ground truth directory.
        iou_thresh (float, optional): IoU threshold for evaluation. Defaults to 0.5.

    Returns:
        list: List of AP values for [Easy, Medium, Hard] settings.
    """
    (
        facebox_list,
        event_list,
        file_list,
        hard_gt_list,
        medium_gt_list,
        easy_gt_list,
    ) = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ["easy", "medium", "hard"]
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2), dtype=np.float32)
        # [hard, medium, easy]
        pbar = tqdm(range(event_num))
        cum_num_image = 0
        for i in pbar:
            pbar.set_description("Processing {}".format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]
            cum_num_image += len(img_list)
            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]
                gt_boxes = np.array(gt_bbx_list[j][0], dtype=np.float32)
                keep_index = np.array(sub_gt_list[j][0], dtype=np.int64)
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_recall, proposal_list = image_eval(
                    pred_info, gt_boxes, ignore, iou_thresh
                )

                _img_pr_info = img_pr_info(
                    thresh_num, pred_info, proposal_list, pred_recall
                )
                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    print("==================== Results ====================")
    print(f"Easy   Val AP: {aps[0]}")
    print(f"Medium Val AP: {aps[1]}")
    print(f"Hard   Val AP: {aps[2]}")
    print("=================================================")

    return aps
