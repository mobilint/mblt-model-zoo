"""Evaluation script for DOTAv1 oriented bounding boxes."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
from tqdm import tqdm

from ..datasets import CustomDOTAv1, get_dota_loader, get_dotav1_class_num, get_dotav1_label
from ..postprocess.common import RatioPad, batch_probiou, rotated_nms, xywhr2xyxyxyxy, xyxyxyxy2xywhr

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine
    from ..results import Results

DOTAV1_CLASS_TO_IDX = {get_dotav1_label(index): index for index in range(get_dotav1_class_num())}


class DOTAResult(NamedTuple):
    """DOTAv1 rotated detection metrics."""

    map50: float
    map5095: float


def _label_to_index(label: str) -> int:
    """Convert a DOTAv1 class token to a class index."""
    try:
        return int(label)
    except ValueError:
        return DOTAV1_CLASS_TO_IDX[label]


def _load_ground_truths(data_path: str, dataset: CustomDOTAv1) -> dict[str, dict[str, torch.Tensor]]:
    """Load DOTAv1 OBB ground-truth labels in original-image coordinates.

    Args:
        data_path: DOTAv1 root directory.
        dataset: Dataset containing image IDs and image paths.

    Returns:
        Mapping from image ID to tensors for classes, polygons, and ``xywhr`` boxes.
    """
    label_dir = Path(data_path) / "labels" / "val"
    original_label_dir = Path(data_path) / "labels" / "val_original"
    ground_truths: dict[str, dict[str, torch.Tensor]] = {}
    for image_id, image_path in zip(dataset.ids, dataset.image_paths):
        image = dataset._load_image(image_path)
        height, width = image.shape[:2]
        label_path = label_dir / f"{image_id}.txt"
        original_label_path = original_label_dir / f"{image_id}.txt"
        classes = []
        polygons = []

        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 9:
                    continue
                cls = _label_to_index(parts[0])
                if cls >= get_dotav1_class_num():
                    raise ValueError(f"Unsupported DOTAv1 class index {cls} in {label_path}.")
                coords = torch.tensor([float(value) for value in parts[1:9]], dtype=torch.float32).reshape(4, 2)
                if coords.numel() and float(coords.max()) <= 1.5:
                    coords[:, 0] *= width
                    coords[:, 1] *= height
                classes.append(cls)
                polygons.append(coords)
        elif original_label_path.is_file():
            for line in original_label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 9:
                    continue
                cls = _label_to_index(parts[8])
                if cls >= get_dotav1_class_num():
                    raise ValueError(f"Unsupported DOTAv1 class index {cls} in {original_label_path}.")
                coords = torch.tensor([float(value) for value in parts[:8]], dtype=torch.float32).reshape(4, 2)
                classes.append(cls)
                polygons.append(coords)

        polygon_tensor = (
            torch.stack(polygons).to(torch.float32) if polygons else torch.zeros((0, 4, 2), dtype=torch.float32)
        )
        boxes = xyxyxyxy2xywhr(polygon_tensor).to(torch.float32) if polygon_tensor.numel() else torch.zeros((0, 5))

        ground_truths[image_id] = {
            "cls": torch.tensor(classes, dtype=torch.int64),
            "polygons": polygon_tensor,
            "bboxes": boxes,
        }
    return ground_truths


def format_dota_results(
    nms_outs: Results,
    input_shape: tuple[int, ...],
    org_shape: tuple[int, ...],
    ratio_pad: list[Any],
    image_ids: tuple[str, ...],
    postprocess: Any,
) -> list[dict[str, Any]]:
    """Format model outputs for DOTAv1 evaluation and export.

    Args:
        nms_outs: Postprocessed model results.
        input_shape: Preprocessed image shape.
        org_shape: Original image shapes.
        ratio_pad: Letterbox metadata.
        image_ids: DOTAv1 image IDs.
        postprocess: Postprocessor instance.

    Returns:
        List of formatted prediction dictionaries.
    """
    labels_list, polygons_list, scores_list, xywhr_list = postprocess.nmsout2eval(
        nms_outs.output,
        input_shape,
        org_shape,
        ratio_pad=ratio_pad,
        include_xywhr=True,
    )
    results = []
    for image_id, labels, polygons, scores, xywhrs in zip(
        image_ids, labels_list, polygons_list, scores_list, xywhr_list
    ):
        for label, polygon, score, xywhr in zip(labels, polygons, scores, xywhrs):
            results.append(
                {
                    "image_id": image_id,
                    "category_id": DOTAV1_CLASS_TO_IDX[label],
                    "category_name": label,
                    "poly": polygon,
                    "score": score,
                    "rbox": xywhr,
                }
            )
    return results


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute AP from recall and precision curves with Ultralytics interpolation."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    grid = np.linspace(0, 1, 101)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(np.interp(grid, mrec, mpre), grid)), mpre, mrec


def _ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16,
) -> np.ndarray:
    """Compute AP per class using the Ultralytics object-detection metric policy."""
    if target_cls.size == 0:
        niou = tp.shape[1] if tp.ndim == 2 else 10
        return np.zeros((0, niou), dtype=np.float64)

    order = np.argsort(-conf)
    tp = tp[order]
    conf = conf[order]
    pred_cls = pred_cls[order]

    unique_classes, target_count = np.unique(target_cls, return_counts=True)
    ap = np.zeros((unique_classes.shape[0], tp.shape[1]), dtype=np.float64)
    for class_index, class_id in enumerate(unique_classes):
        pred_mask = pred_cls == class_id
        num_labels = target_count[class_index]
        num_predictions = pred_mask.sum()
        if num_predictions == 0 or num_labels == 0:
            continue

        false_positive = (1 - tp[pred_mask]).cumsum(0)
        true_positive = tp[pred_mask].cumsum(0)
        recall = true_positive / (num_labels + eps)
        precision = true_positive / (true_positive + false_positive + eps)
        for iou_index in range(tp.shape[1]):
            ap[class_index, iou_index], _, _ = _compute_ap(recall[:, iou_index], precision[:, iou_index])
    return ap


def _match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
) -> np.ndarray:
    """Match predictions to ground-truth boxes with Ultralytics one-to-one matching."""
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    if pred_classes.numel() == 0 or true_classes.numel() == 0:
        return correct

    correct_class = true_classes[:, None] == pred_classes
    iou_np = (iou * correct_class).cpu().numpy()
    for iou_index, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.array(np.nonzero(iou_np >= threshold)).T
        if matches.shape[0] == 0:
            continue
        if matches.shape[0] > 1:
            matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), iou_index] = True
    return correct


def _empty_stats() -> dict[str, list[np.ndarray]]:
    """Create an empty DOTAv1 metric statistics accumulator."""
    return {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}


def _append_stats(stats: dict[str, list[np.ndarray]], image_stats: dict[str, np.ndarray]) -> None:
    """Append one image's metric statistics to the accumulator."""
    for key, value in image_stats.items():
        stats[key].append(value)


def _nms_output_to_predictions(nms_out: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert OBB NMS rows to the prediction dictionary used by metric matching."""
    if nms_out.numel() == 0:
        return {
            "bboxes": torch.zeros((0, 5), dtype=torch.float32),
            "conf": torch.zeros(0, dtype=torch.float32),
            "cls": torch.zeros(0, dtype=torch.int64),
        }

    nms_out = nms_out.detach().cpu()
    return {
        "bboxes": torch.cat([nms_out[:, :4], nms_out[:, 6:7]], dim=-1).to(torch.float32),
        "conf": nms_out[:, 4].to(torch.float32),
        "cls": nms_out[:, 5].to(torch.int64),
    }


def _ratio_pad_for_shape(
    input_shape: tuple[int, int],
    org_shape: tuple[int, int],
    ratio_pad: RatioPad | None,
) -> tuple[float, tuple[float, float]]:
    """Return letterbox gain and padding for an image."""
    if ratio_pad is not None:
        return float(ratio_pad[0][0]), (float(ratio_pad[1][0]), float(ratio_pad[1][1]))

    gain = min(input_shape[0] / org_shape[0], input_shape[1] / org_shape[1])
    pad_x = round((input_shape[1] - round(org_shape[1] * gain)) / 2 - 0.1)
    pad_y = round((input_shape[0] - round(org_shape[0] * gain)) / 2 - 0.1)
    return float(gain), (float(pad_x), float(pad_y))


def _ground_truth_to_input_space(
    ground_truth: dict[str, torch.Tensor],
    input_shape: tuple[int, int],
    org_shape: tuple[int, int],
    ratio_pad: RatioPad | None,
) -> dict[str, torch.Tensor]:
    """Transform original-image DOTAv1 polygons to letterboxed ``xywhr`` boxes."""
    classes = ground_truth["cls"]
    polygons = ground_truth.get("polygons")
    if polygons is None:
        return {"cls": classes, "bboxes": ground_truth["bboxes"]}
    if polygons.numel() == 0:
        return {"cls": classes, "bboxes": torch.zeros((0, 5), dtype=torch.float32)}

    gain, pad = _ratio_pad_for_shape(input_shape, org_shape, ratio_pad)
    transformed = polygons.clone().to(torch.float32)
    transformed[..., 0] = transformed[..., 0] * gain + pad[0]
    transformed[..., 1] = transformed[..., 1] * gain + pad[1]
    return {"cls": classes, "bboxes": xyxyxyxy2xywhr(transformed).to(torch.float32)}


def _process_image_stats(
    predictions: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    iouv: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Build one image's true-positive, confidence, class, and target arrays."""
    target_cls = target["cls"].cpu().numpy()
    if target["cls"].numel() == 0 or predictions["cls"].numel() == 0:
        true_positive = np.zeros((predictions["cls"].shape[0], iouv.numel()), dtype=bool)
    else:
        iou = batch_probiou(target["bboxes"], predictions["bboxes"])
        true_positive = _match_predictions(predictions["cls"], target["cls"], iou, iouv)
    return {
        "tp": true_positive,
        "conf": predictions["conf"].cpu().numpy(),
        "pred_cls": predictions["cls"].cpu().numpy(),
        "target_cls": target_cls,
    }


def _evaluate_stats(stats: dict[str, list[np.ndarray]], niou: int = 10) -> DOTAResult:
    """Compute DOTAv1 mAP metrics from accumulated validator statistics."""
    target_cls = np.concatenate(stats["target_cls"], 0) if stats["target_cls"] else np.zeros(0, dtype=np.float64)
    if target_cls.size == 0:
        return DOTAResult(map50=0.0, map5095=0.0)

    tp = np.concatenate(stats["tp"], 0) if stats["tp"] else np.zeros((0, niou), dtype=bool)
    conf = np.concatenate(stats["conf"], 0) if stats["conf"] else np.zeros(0, dtype=np.float64)
    pred_cls = np.concatenate(stats["pred_cls"], 0) if stats["pred_cls"] else np.zeros(0, dtype=np.float64)
    ap = _ap_per_class(tp, conf, pred_cls, target_cls)
    if ap.size == 0:
        return DOTAResult(map50=0.0, map5095=0.0)
    return DOTAResult(map50=float(ap[:, 0].mean()), map5095=float(ap.mean()))


def evaluate_dota_predictions(
    ground_truths: dict[str, dict[str, torch.Tensor]],
    predictions: list[dict[str, Any]],
) -> DOTAResult:
    """Evaluate DOTAv1 predictions with local rotated mAP.

    Args:
        ground_truths: Mapping of image IDs to class and OBB tensors.
        predictions: Formatted prediction dictionaries.

    Returns:
        Rotated mAP at IoU ``0.50`` and averaged across ``0.50:0.95``.
    """
    iouv = torch.linspace(0.5, 0.95, 10)
    stats = _empty_stats()
    predictions_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_image[str(prediction["image_id"])].append(prediction)

    image_ids = set(ground_truths) | set(predictions_by_image)
    for image_id in sorted(image_ids):
        rows = predictions_by_image.get(image_id, [])
        if rows:
            pred_dict = {
                "bboxes": torch.tensor([row["rbox"] for row in rows], dtype=torch.float32),
                "conf": torch.tensor([row["score"] for row in rows], dtype=torch.float32),
                "cls": torch.tensor([row["category_id"] for row in rows], dtype=torch.int64),
            }
        else:
            pred_dict = _nms_output_to_predictions(torch.zeros((0, 7), dtype=torch.float32))
        target = ground_truths.get(
            image_id,
            {"cls": torch.zeros(0, dtype=torch.int64), "bboxes": torch.zeros((0, 5), dtype=torch.float32)},
        )
        _append_stats(stats, _process_image_stats(pred_dict, target, iouv))
    return _evaluate_stats(stats, niou=iouv.numel())


def save_dota_task1_predictions(predictions: list[dict[str, Any]], save_dir: str) -> tuple[Path, Path]:
    """Save split and merged predictions in DOTA Task1 text format.

    Args:
        predictions: Formatted prediction dictionaries.
        save_dir: Directory where prediction folders are written.

    Returns:
        Tuple of split and merged prediction directories.
    """
    root = Path(save_dir)
    pred_txt = root / "predictions_txt"
    pred_merged_txt = root / "predictions_merged_txt"
    pred_txt.mkdir(parents=True, exist_ok=True)
    pred_merged_txt.mkdir(parents=True, exist_ok=True)

    for cls_idx in range(get_dotav1_class_num()):
        (pred_txt / f"Task1_{get_dotav1_label(cls_idx)}.txt").write_text("", encoding="utf-8")
        (pred_merged_txt / f"Task1_{get_dotav1_label(cls_idx)}.txt").write_text("", encoding="utf-8")

    for pred in predictions:
        class_name = pred["category_name"]
        polygon = pred["poly"]
        with (pred_txt / f"Task1_{class_name}.txt").open("a", encoding="utf-8") as file:
            file.write(
                f"{pred['image_id']} {pred['score']} "
                f"{polygon[0]} {polygon[1]} {polygon[2]} {polygon[3]} "
                f"{polygon[4]} {polygon[5]} {polygon[6]} {polygon[7]}\n"
            )

    merged_results: dict[str, list[list[float]]] = defaultdict(list)
    offset_pattern = re.compile(r"(\d+)___(\d+)")
    for pred in predictions:
        image_id = pred["image_id"]
        base_image_id = image_id.split("__", 1)[0]
        offset_match = offset_pattern.search(image_id)
        x_offset, y_offset = (0, 0)
        if offset_match is not None:
            x_offset, y_offset = (int(offset_match.group(1)), int(offset_match.group(2)))
        rbox = list(pred["rbox"])
        rbox[0] += x_offset
        rbox[1] += y_offset
        merged_results[base_image_id].append([*rbox, pred["score"], float(pred["category_id"])])

    for image_id, rows in merged_results.items():
        bbox = torch.tensor(rows, dtype=torch.float32)
        if bbox.numel() == 0:
            continue
        max_wh = max(float(torch.max(bbox[:, :2]).item() * 2), 1.0)
        class_offsets = bbox[:, 6:7] * max_wh
        boxes = bbox[:, :5].clone()
        boxes[:, :2] += class_offsets
        keep = rotated_nms(boxes, bbox[:, 5], 0.3)
        bbox = bbox[keep]
        polygons = xywhr2xyxyxyxy(bbox[:, :5]).reshape(-1, 8)
        for polygon, score, cls in zip(polygons.tolist(), bbox[:, 5].tolist(), bbox[:, 6].tolist()):
            class_name = get_dotav1_label(int(cls))
            rounded_polygon = [round(float(value), 3) for value in polygon]
            with (pred_merged_txt / f"Task1_{class_name}.txt").open("a", encoding="utf-8") as file:
                file.write(
                    f"{image_id} {round(float(score), 3)} "
                    f"{rounded_polygon[0]} {rounded_polygon[1]} {rounded_polygon[2]} {rounded_polygon[3]} "
                    f"{rounded_polygon[4]} {rounded_polygon[5]} {rounded_polygon[6]} {rounded_polygon[7]}\n"
                )

    return pred_txt, pred_merged_txt


def _nms_output_list(nms_output: Any) -> list[torch.Tensor]:
    """Normalize postprocess output to a per-image list of OBB tensors."""
    if isinstance(nms_output, list):
        return nms_output
    if isinstance(nms_output, tuple):
        return list(nms_output)
    if isinstance(nms_output, torch.Tensor):
        if nms_output.ndim == 3:
            return [image[image[:, 4] > 0] for image in nms_output]
        return [nms_output]
    raise TypeError(f"Unsupported OBB NMS output type: {type(nms_output).__name__}.")


def eval_dota(
    model: MBLT_Engine,
    data_path: str,
    batch_size: int,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
    save_dir: str | None = None,
) -> DOTAResult:
    """Evaluate a model on DOTAv1 validation.

    Args:
        model: Model engine to evaluate.
        data_path: DOTAv1 dataset root.
        batch_size: Batch size for evaluation.
        conf_thres: Optional confidence threshold override.
        iou_thres: Optional IoU threshold override.
        save_dir: Optional directory for DOTA Task1 prediction files.

    Returns:
        Local rotated mAP scores.
    """
    if model.post_cfg["task"] != "obb":
        raise NotImplementedError(f"Task {model.post_cfg['task']} is not supported for DOTAv1 evaluation.")

    dataset = CustomDOTAv1(data_path)
    dataloader = get_dota_loader(dataset, batch_size, model.preprocess_with_metadata)
    model.set_postprocess_thresholds(conf_thres=conf_thres, iou_thres=iou_thres)
    ground_truths = _load_ground_truths(data_path, dataset)
    iouv = torch.linspace(0.5, 0.95, 10)
    stats = _empty_stats()

    results = []
    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter, desc="Evaluating DOTAv1")
    inference_time = 0.0
    cum_num_data = 0

    for input_npu, org_shape, ratio_pad, image_ids in pbar:
        cum_num_data += len(image_ids)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic
        nms_outs = model.postprocess(out_npu)
        input_shape = tuple(int(value) for value in input_npu.shape[1:-1])
        for nms_out, image_id, image_shape, image_ratio_pad in zip(
            _nms_output_list(nms_outs.output),
            image_ids,
            org_shape,
            ratio_pad,
        ):
            target = _ground_truth_to_input_space(
                ground_truths[image_id],
                input_shape,
                (int(image_shape[0]), int(image_shape[1])),
                image_ratio_pad,
            )
            _append_stats(stats, _process_image_stats(_nms_output_to_predictions(nms_out), target, iouv))
        if save_dir is not None:
            results.extend(
                format_dota_results(
                    nms_outs,
                    input_shape,
                    org_shape,
                    ratio_pad,
                    image_ids,
                    model.postprocessor,
                )
            )
        pbar.set_postfix_str(f"NPU FPS: {cum_num_data / inference_time:.3f}")

    pbar.close()
    map_score = _evaluate_stats(stats, niou=iouv.numel())
    if save_dir is not None:
        save_dota_task1_predictions(results, save_dir)
    print("DOTAv1 evaluation completed")
    return map_score
