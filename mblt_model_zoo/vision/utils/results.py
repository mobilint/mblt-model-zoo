"""
Results processing and plotting.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
import torch
from PIL import Image

from .datasets import (
    get_ade20k_palette,
    get_coco_det_palette,
    get_coco_keypoint_palette,
    get_coco_label,
    get_coco_limb_palette,
    get_coco_pose_skeleton,
    get_dotav1_label,
    get_dotav1_palette,
    get_imagenet_label,
)
from .postprocess.common import crop_mask, scale_boxes, scale_coords, scale_masks, scale_rboxes, xywhr2xyxyxyxy
from .types import ListTensorLike, NestedListTensorLike, TensorLike

LW = 2  # line width
RADIUS = 5  # circle radius
ALPHA = 0.3  # alpha for overlay
DENSE_OVERLAY_ALPHA = 0.7


class Results:
    """
    Class for handling and plotting model inference results.
    Class for handling, processing, and plotting model inference results.
    Provides methods to visualize detections, masks, and keypoints on images.
    """

    def __init__(
        self,
        pre_cfg: dict,
        post_cfg: dict,
        output: TensorLike | ListTensorLike | NestedListTensorLike,
        **kwargs,
    ) -> None:
        """
        Initializes the Results object.
        Args:
            pre_cfg (dict): Preprocessing configuration.
            post_cfg (dict): Postprocessing configuration.
            output (TensorLike | ListTensorLike | NestedListTensorLike): Raw model output.
            **kwargs: Additional arguments.
        """
        self.pre_cfg = pre_cfg
        self.post_cfg = post_cfg
        self.task = post_cfg["task"]
        self.conf_thres = kwargs.get("conf_thres", 0.25)
        self.acc: torch.Tensor | np.ndarray | None = None
        self.box_cls: torch.Tensor | np.ndarray | None = None
        self.mask: torch.Tensor | np.ndarray | None = None
        self.depth: torch.Tensor | np.ndarray | list[TensorLike] | None = None
        self.semantic_mask: torch.Tensor | np.ndarray | None = None
        self.output: TensorLike | ListTensorLike | NestedListTensorLike | None = None
        self.labels: torch.Tensor | None = None
        self.scores: torch.Tensor | None = None
        self.boxes: torch.Tensor | None = None
        self.rboxes: torch.Tensor | None = None
        self.kpts: torch.Tensor | None = None
        self.set_output(output)

    def _read_image(self, source_path: str | np.ndarray | Image.Image) -> np.ndarray:
        """
        Internal method to read an image from various input types and convert to BGR format.
        Args:
            source_path (str | np.ndarray | Image.Image): Path to image or image object.
        Returns:
            np.ndarray: Image in BGR format (cv2 style).
        """
        source_img = None
        if isinstance(source_path, Image.Image):  # PIL image open
            source_img = source_path.convert("RGB")
            source_img = np.array(source_img)
            source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
        elif isinstance(source_path, np.ndarray):
            source_img = np.array(source_path)  # assume imread or video read is made in BGR format
            assert source_img.shape[2] == 3, f"Got unexpected shape for source_img={source_img.shape}."
        else:  # str image path
            assert os.path.exists(source_path) and os.path.isfile(source_path), (
                f"File {source_path} does not exist or is not a file."
            )
            source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if source_img is None:
            raise ValueError(f"Failed to read image from {type(source_path)}.")
        return source_img

    def set_output(self, output: TensorLike | ListTensorLike | NestedListTensorLike) -> None:
        """
        Sets variables from the raw model output based on the task.
        Args:
            output (TensorLike | ListTensorLike | NestedListTensorLike): Raw model output.
        Raises:
            NotImplementedError: If the task is not supported.
        """
        self.acc = None
        self.box_cls = None
        self.mask = None
        self.depth = None
        self.semantic_mask = None
        if self.task.lower() == "image_classification":
            if isinstance(output, Sequence):
                raise TypeError(f"Expected tensor output for task {self.task}, got {type(output)}.")
            self.acc = output
        elif self.task.lower() in {"object_detection", "face_detection", "pose_estimation", "obb"}:
            if not isinstance(output, Sequence):
                raise TypeError(f"Expected list output for task {self.task}, got {type(output)}.")
            self.box_cls = cast(TensorLike, output[0])
        elif self.task.lower() == "instance_segmentation":
            if not isinstance(output, Sequence) or not isinstance(output[0], Sequence):
                raise TypeError(f"Expected nested list output for task {self.task}, got {type(output)}.")
            seg_output = cast(ListTensorLike, output[0])
            self.box_cls = cast(TensorLike, seg_output[0])
            self.mask = cast(TensorLike, seg_output[1])
        elif self.task.lower() == "depth_estimation":
            if isinstance(output, Sequence) and not isinstance(output, (np.ndarray, torch.Tensor)):
                if not all(isinstance(item, (np.ndarray, torch.Tensor)) for item in output):
                    raise TypeError(f"Expected depth-map tensors for task {self.task}, got {type(output)}.")
                self.depth = [cast(TensorLike, item) for item in output]
            elif isinstance(output, (np.ndarray, torch.Tensor)):
                self.depth = output
            else:
                raise TypeError(f"Expected tensor depth output for task {self.task}, got {type(output)}.")
        elif self.task.lower() == "semantic_segmentation":
            if not isinstance(output, (np.ndarray, torch.Tensor)):
                raise TypeError(f"Expected tensor semantic output for task {self.task}, got {type(output)}.")
            self.semantic_mask = output
        else:
            raise NotImplementedError(f"Task {self.task} is not supported for plotting results.")
        self.output = output  # store raw output

    def plot(
        self,
        source_path: str | np.ndarray | Image.Image,
        save_path: str | None = None,
        **kwargs,
    ) -> np.ndarray | None:
        """
        Plots the results on the source image.
        Plots the inference results on the source image.
        Args:
            source_path (str | np.ndarray | Image.Image): Path or image object.
            save_path (str, optional): Path to save the plotted image. Defaults to None.
            **kwargs: Additional arguments.
            source_path (str | np.ndarray | Image.Image): The image to plot on.
            save_path (str, optional): If provided, the result image will be saved to this path.
            **kwargs: Additional task-specific plotting options (e.g., topk for classification).
        Returns:
            np.ndarray: The image with results visualized (in BGR format).
        Raises:
            NotImplementedError: If the task is not supported.
            NotImplementedError: If the task is not supported for plotting.
        """
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.task.lower() == "image_classification":
            return self._plot_image_classification(source_path, save_path, **kwargs)
        elif self.task.lower() in {"object_detection", "face_detection"}:
            return self._plot_object_detection(source_path, save_path, **kwargs)
        elif self.task.lower() == "instance_segmentation":
            return self._plot_instance_segmentation(source_path, save_path, **kwargs)
        elif self.task.lower() == "depth_estimation":
            return self._plot_depth_estimation(source_path, save_path, **kwargs)
        elif self.task.lower() == "semantic_segmentation":
            return self._plot_semantic_segmentation(source_path, save_path, **kwargs)
        elif self.task.lower() == "pose_estimation":
            return self._plot_pose_estimation(source_path, save_path, **kwargs)
        elif self.task.lower() == "obb":
            return self._plot_obb(source_path, save_path, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task} is not supported for plotting results.")

    def _plot_depth_estimation(
        self,
        source_path: str | np.ndarray | Image.Image,
        save_path: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Colorize the first depth map with near objects in red and blend it over the original image."""

        del kwargs
        if self.depth is None:
            raise ValueError("No depth output found.")
        depth_value = self.depth[0] if isinstance(self.depth, list) else self.depth
        depth = depth_value.detach().cpu().numpy() if isinstance(depth_value, torch.Tensor) else depth_value
        if depth.ndim == 3:
            depth = depth[0]
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D depth map or [B, H, W], got {depth.shape}.")
        image = self._read_image(source_path)
        image_height, image_width = image.shape[:2]
        depth = self._restore_depth_map(depth, (image_height, image_width))
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise ValueError("Depth output contains no positive finite values.")
        disparity = np.zeros(depth.shape, dtype=np.float32)
        disparity[valid] = 1.0 / depth[valid]
        lower, upper = np.percentile(disparity[valid], (2, 98))
        if upper <= lower:
            upper = lower + 1e-6
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        normalized[valid] = np.clip((disparity[valid] - lower) * 255 / (upper - lower), 0, 255).astype(np.uint8)
        overlay = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        overlay[~valid] = 0
        result = cv2.addWeighted(image, 1.0 - DENSE_OVERLAY_ALPHA, overlay, DENSE_OVERLAY_ALPHA, 0)
        if save_path is not None:
            cv2.imwrite(save_path, result)
        return result

    def _plot_semantic_segmentation(
        self,
        source_path: str | np.ndarray | Image.Image,
        save_path: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Colorize a semantic class map and blend it over the original image."""

        del kwargs
        if self.semantic_mask is None:
            raise ValueError("No semantic output found.")
        class_map = (
            self.semantic_mask.detach().cpu().numpy()
            if isinstance(self.semantic_mask, torch.Tensor)
            else self.semantic_mask
        )
        if class_map.ndim == 3:
            class_map = class_map[0]
        if class_map.ndim != 2:
            raise ValueError(f"Expected a 2D semantic map or [B, H, W], got {class_map.shape}.")
        image = self._read_image(source_path)
        image_shape = (int(image.shape[0]), int(image.shape[1]))
        class_map = self._restore_semantic_map(class_map, image_shape)
        nc = int(self.post_cfg.get("nc", 150 if self.post_cfg.get("dataset") == "ade20k" else 19))
        if class_map.size and (int(class_map.min()) < 0 or int(class_map.max()) >= nc):
            raise ValueError(f"Semantic class-map values must be in [0, {nc - 1}].")
        palette = np.array([get_ade20k_palette(index) for index in range(nc)], dtype=np.uint8)
        overlay = palette[class_map.astype(np.int64)]
        result = cv2.addWeighted(image, 1.0 - DENSE_OVERLAY_ALPHA, overlay, DENSE_OVERLAY_ALPHA, 0)
        if save_path is not None:
            cv2.imwrite(save_path, result)
        return result

    def _restore_semantic_map(self, class_map: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        """Undo the configured letterbox transform using nearest-neighbor interpolation."""

        letterbox_cfg = self.pre_cfg.get("LetterBox", {})
        input_shape = letterbox_cfg.get("img_size")
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            return cv2.resize(class_map, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        input_height, input_width = int(input_shape[0]), int(input_shape[1])
        image_height, image_width = image_shape
        ratio = min(input_height / image_height, input_width / image_width)
        unpadded_width = int(round(image_width * ratio))
        unpadded_height = int(round(image_height * ratio))
        left = int(round((input_width - unpadded_width) / 2 - 0.1))
        top = int(round((input_height - unpadded_height) / 2 - 0.1))
        scale_x, scale_y = class_map.shape[1] / input_width, class_map.shape[0] / input_height
        cropped = class_map[
            int(round(top * scale_y)) : int(round((top + unpadded_height) * scale_y)),
            int(round(left * scale_x)) : int(round((left + unpadded_width) * scale_x)),
        ]
        if cropped.size == 0:
            raise ValueError("Semantic letterbox restoration produced an empty crop.")
        return cv2.resize(cropped, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

    def _restore_depth_map(self, depth: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        """Undo the configured letterbox transform and resize a depth map to an image."""

        letterbox_cfg = self.pre_cfg.get("LetterBox", {})
        input_shape = letterbox_cfg.get("img_size")
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            return cv2.resize(depth, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
        input_height, input_width = int(input_shape[0]), int(input_shape[1])
        image_height, image_width = image_shape
        ratio = min(input_height / image_height, input_width / image_width)
        unpadded_width, unpadded_height = int(round(image_width * ratio)), int(round(image_height * ratio))
        left = int(round((input_width - unpadded_width) / 2 - 0.1))
        top = int(round((input_height - unpadded_height) / 2 - 0.1))
        scale_x, scale_y = depth.shape[1] / input_width, depth.shape[0] / input_height
        cropped = depth[
            int(round(top * scale_y)) : int(round((top + unpadded_height) * scale_y)),
            int(round(left * scale_x)) : int(round((left + unpadded_width) * scale_x)),
        ]
        if cropped.size == 0:
            raise ValueError("Depth letterbox restoration produced an empty crop.")
        return cv2.resize(cropped, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

    def _plot_image_classification(
        self,
        source_path: str | np.ndarray | Image.Image | None = None,
        save_path: str | None = None,
        topk: int = 5,
        **kwargs,
    ) -> np.ndarray | None:
        assert self.acc is not None, "No accuracy output found."
        if isinstance(self.acc, np.ndarray):
            self.acc = torch.tensor(self.acc)
        topk_probs, topk_indices = torch.topk(self.acc, topk)
        topk_probs = np.atleast_1d(topk_probs.squeeze().numpy())
        topk_indices = np.atleast_1d(topk_indices.squeeze().numpy())
        # load labels
        labels = [get_imagenet_label(i) for i in topk_indices]
        comments = []
        for i in range(topk):
            comments.append(f"{labels[i]}: {topk_probs[i] * 100:.2f}%")
            print(f"Label: {labels[i]}, Probability: {topk_probs[i] * 100:.2f}%")
        if source_path is not None and save_path is not None:
            comments_str = "\n".join(comments)
            img = self._read_image(source_path)
            avg_color = img.mean(axis=(0, 1))
            txt_color = (
                int(255 - avg_color[0]),
                int(255 - avg_color[1]),
                int(255 - avg_color[2]),
            )
            for i, line in enumerate(comments_str.splitlines()):
                (_, h), _ = cv2.getTextSize(
                    text=line,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1,
                )
                img = cv2.putText(
                    img,
                    line,
                    (15, 15 + int(1.5 * i * h)),  # line spacing
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=txt_color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                cv2.imwrite(save_path, img)
            return img
        else:
            return None

    def _plot_object_detection(
        self,
        source_path: str | np.ndarray | Image.Image,
        save_path: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        box_cls = self._box_cls_tensor()
        assert box_cls.shape[1] == 6 + self.post_cfg.get("n_extra", 0), (
            f"Got unexpected shape for object detection box_cls={box_cls.shape}."
        )
        img = self._read_image(source_path)
        img1_shape = cast(tuple[int, int], self.pre_cfg["LetterBox"]["img_size"])
        img0_shape: tuple[int, int] = (img.shape[0], img.shape[1])
        self.labels = box_cls[:, 5].to(torch.int64)
        self.scores = box_cls[:, 4]
        self.boxes = scale_boxes(
            img1_shape,
            box_cls[:, :4],
            img0_shape,
        )
        boxes = self.boxes
        scores = self.scores
        labels = self.labels
        contours: dict[int, list[np.ndarray]] = {}
        for box, score, label in zip(boxes, scores, labels):
            label_idx = int(label.item())
            palette = self._get_detection_palette(label_idx)
            img = cv2.putText(
                img,
                f"{self._get_detection_label(label_idx)} {int(100 * score)}%",
                (int(box[0]), int(box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                palette,
                1,
                cv2.LINE_AA,
            )
            contours.setdefault(label_idx, []).append(
                np.array(
                    [
                        [int(box[0]), int(box[1])],
                        [int(box[2]), int(box[1])],
                        [int(box[2]), int(box[3])],
                        [int(box[0]), int(box[3])],
                    ]
                )
            )
        for label, contour in contours.items():
            if len(contour) > 0:
                cv2.drawContours(
                    img,
                    contour,
                    -1,
                    self._get_detection_palette(label),
                    LW,
                )
        if save_path is not None:
            cv2.imwrite(save_path, img)
        return img

    def _plot_instance_segmentation(
        self,
        source_path: str | np.ndarray | Image.Image,
        save_path: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        img = self._plot_object_detection(source_path, None, **kwargs)
        assert self.mask is not None, "No mask output found."
        assert self.boxes is not None, "No boxes output found."
        assert self.labels is not None, "No labels output found."
        mask = self._mask_tensor()
        img0_shape: tuple[int, int] = (img.shape[0], img.shape[1])
        masks = (
            crop_mask(scale_masks(mask, img0_shape), self.boxes)
            .gt_(0.0)
            .permute(1, 2, 0)
            .to(torch.float32)
            .cpu()
            .numpy()
        )
        overlay = np.zeros((masks.shape[0], masks.shape[1], 3))
        for i, label in enumerate(self.labels):
            label_idx = int(label.item())
            overlay = np.maximum(
                overlay,
                masks[:, :, i][:, :, np.newaxis] * np.array(get_coco_det_palette(label_idx)).reshape(1, 1, 3),
            )
        total_mask = overlay.max(axis=2, keepdims=True)
        inv_mask = 1 - ALPHA * total_mask / 255
        img = (img * inv_mask + overlay * ALPHA).astype(np.uint8)
        if save_path is not None:
            cv2.imwrite(save_path, img)
        return img

    def _plot_pose_estimation(
        self,
        source_path: str | np.ndarray | Image.Image,
        save_path: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        img = self._plot_object_detection(source_path, None, **kwargs)
        box_cls = self._box_cls_tensor()
        img0_shape: tuple[int, int] = (img.shape[0], img.shape[1])
        self.kpts = scale_coords(
            self.pre_cfg["LetterBox"]["img_size"],
            box_cls[:, 6:].reshape(-1, 17, 3),
            img0_shape,
        )
        kpts = self.kpts
        if kpts is None:
            raise ValueError("No keypoints output found.")
        for kpt in kpts:
            for i, (x, y, v) in enumerate(kpt):
                color_k = get_coco_keypoint_palette(i)
                # if v < self.conf_thres:
                #    continue
                cv2.circle(
                    img,
                    (int(x), int(y)),
                    RADIUS,
                    color_k,
                    -1,
                    lineType=cv2.LINE_AA,
                )
            for j, sk in enumerate(get_coco_pose_skeleton()):
                pos1 = (int(kpt[sk[0] - 1, 0]), int(kpt[sk[0] - 1, 1]))
                pos2 = (int(kpt[sk[1] - 1, 0]), int(kpt[sk[1] - 1, 1]))
                # conf1 = kpt[sk[0] - 1, 2]
                # conf2 = kpt[sk[1] - 1, 2]
                # if conf1 < self.conf_thres or conf2 < self.conf_thres:
                #    continue
                cv2.line(
                    img,
                    pos1,
                    pos2,
                    get_coco_limb_palette(j),
                    thickness=int(np.ceil(LW / 2)),
                    lineType=cv2.LINE_AA,
                )
        if save_path is not None:
            cv2.imwrite(save_path, img)
        return img

    def _plot_obb(
        self,
        source_path: str | np.ndarray | Image.Image,
        save_path: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Plots oriented bounding boxes on an image.

        Args:
            source_path: Path or image object.
            save_path: Optional path to save the plotted image.
            **kwargs: Additional plotting arguments.

        Returns:
            The plotted BGR image.
        """
        del kwargs
        box_cls = self._box_cls_tensor()
        assert box_cls.shape[1] == 7, f"Got unexpected shape for OBB box_cls={box_cls.shape}."
        img = self._read_image(source_path)
        img0_shape: tuple[int, int] = (img.shape[0], img.shape[1])
        self.labels = box_cls[:, 5].to(torch.int64)
        self.scores = box_cls[:, 4]
        self.rboxes = scale_rboxes(
            self.pre_cfg["LetterBox"]["img_size"],
            torch.cat([box_cls[:, :4], box_cls[:, 6:7]], dim=-1),
            img0_shape,
        )
        polygons = xywhr2xyxyxyxy(self.rboxes).to(torch.int32).cpu().numpy()
        for polygon, score, label in zip(polygons, self.scores, self.labels):
            label_idx = int(label.item())
            color = get_dotav1_palette(label_idx)
            text_anchor = polygon.min(axis=0)
            img = cv2.putText(
                img,
                f"{get_dotav1_label(label_idx)} {int(100 * score)}%",
                (int(text_anchor[0]), int(text_anchor[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.drawContours(img, [polygon.reshape(-1, 1, 2)], -1, color, LW)
        if save_path is not None:
            cv2.imwrite(save_path, img)
        return img

    def _box_cls_tensor(self) -> torch.Tensor:
        """Returns detection output as a torch tensor."""
        if self.box_cls is None:
            raise ValueError("No box_cls output found.")
        if isinstance(self.box_cls, np.ndarray):
            return torch.from_numpy(self.box_cls)
        return self.box_cls

    def _get_detection_label(self, label_idx: int) -> str:
        """Return the display label for detection-style tasks."""
        if self.task.lower() == "face_detection":
            if label_idx != 0:
                raise ValueError(f"Unexpected face_detection class index: {label_idx}.")
            return "face"
        return get_coco_label(label_idx)

    def _get_detection_palette(self, label_idx: int) -> tuple[int, int, int]:
        """Return the display color for detection-style tasks."""
        if self.task.lower() == "face_detection":
            return get_coco_det_palette(0)
        return get_coco_det_palette(label_idx)

    def _mask_tensor(self) -> torch.Tensor:
        """Returns segmentation mask output as a torch tensor."""
        if self.mask is None:
            raise ValueError("No mask output found.")
        if isinstance(self.mask, np.ndarray):
            return torch.from_numpy(self.mask)
        return self.mask
