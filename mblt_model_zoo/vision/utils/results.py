from typing import Union
import os
import numpy as np
import torch
from PIL import Image
import cv2
from .datasets import *
from .postprocess.common import *
from .types import TensorLike, ListTensorLike

LW = 2  # line width
RADIUS = 5  # circle radius
ALPHA = 0.3  # alpha for overlay
# for drawing bounding box


class Results:
    def __init__(
        self,
        pre_cfg: dict,
        post_cfg: dict,
        output: Union[TensorLike, ListTensorLike],
        **kwargs,
    ):
        self.pre_cfg = pre_cfg
        self.post_cfg = post_cfg
        self.task = post_cfg["task"]
        self.set_output(output)
        self.conf_thres = kwargs.get("conf_thres", 0.25)

    def _read_image(self, source_path: Union[str, cv2.typing.MatLike, Image.Image]):
        source_img = None

        if isinstance(source_path, Image.Image):  # PIL image open
            source_img = source_path.convert("RGB")
            source_img = np.array(source_img)
            source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
        elif isinstance(source_path, cv2.typing.MatLike):
            source_img = np.array(
                source_path
            )  # assume imread or video read is made in BGR format
            assert (
                source_img.shape[2] == 3
            ), f"Got unexpected shape for source_img={source_img.shape}."
        else:  # str image path
            assert os.path.exists(source_path) and os.path.isfile(
                source_path
            ), f"File {source_path} does not exist or is not a file."
            source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)

        return source_img

    def set_output(self, output: Union[TensorLike, ListTensorLike]):
        self.acc = None
        self.box_cls = None
        self.mask = None

        if self.task.lower() == "image_classification":
            if isinstance(output, list):
                assert len(output) == 1, f"Got unexpected output={output}."
                output = output[0]
            self.acc = output
        elif (
            self.task.lower() == "object_detection"
            or self.task.lower() == "pose_estimation"
        ):
            if isinstance(output, list):
                assert len(output) == 1, f"Got unexpected output={output}."
                output = output[0]
            self.box_cls = output
        elif self.task.lower() == "instance_segmentation":
            assert isinstance(
                output, list
            ), f"Got unexpected output={output}. It should be a list."
            if len(output) == 2:  # [box_cls, mask]
                pass
            elif len(output) == 1:  # [[box_cls, mask]]
                assert len(output[0]) == 2, f"Got unexpected output={output}."
                output = output[0]
            else:
                raise ValueError(f"Got unexpected output={output}.")
            self.box_cls = output[0]
            self.mask = output[1]
        else:
            raise NotImplementedError(
                f"Task {self.task} is not supported for plotting results."
            )
        self.output = output  # store raw output

    def plot(
        self,
        source_path: Union[str, cv2.typing.MatLike, Image.Image],
        save_path: str = None,
        **kwargs,
    ):
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if self.task.lower() == "image_classification":
            return self._plot_image_classification(source_path, save_path, **kwargs)
        elif self.task.lower() == "object_detection":
            return self._plot_object_detection(source_path, save_path, **kwargs)
        elif self.task.lower() == "instance_segmentation":
            return self._plot_instance_segmentation(source_path, save_path, **kwargs)
        elif self.task.lower() == "pose_estimation":
            return self._plot_pose_estimation(source_path, save_path, **kwargs)
        else:
            raise NotImplementedError(
                f"Task {self.task} is not supported for plotting results."
            )

    def _plot_image_classification(
        self,
        source_path: Union[str, cv2.typing.MatLike, Image.Image] = None,
        save_path: str = None,
        topk=5,
        **kwargs,
    ):
        assert self.acc is not None, "No accuracy output found."
        if isinstance(self.acc, np.ndarray):
            self.acc = torch.tensor(self.acc)

        topk_probs, topk_indices = torch.topk(self.acc, topk)
        topk_probs = topk_probs.squeeze().numpy()
        topk_indices = topk_indices.squeeze().numpy()

        # load labels
        labels = [get_imagenet_label(i) for i in topk_indices]
        comments = []
        for i in range(topk):
            comments.append(f"{labels[i]}: {topk_probs[i]*100:.2f}%")
            print(f"Label: {labels[i]}, Probability: {topk_probs[i]*100:.2f}%")

        if source_path is not None and save_path is not None:
            comments = "\n".join(comments)
            img = self._read_image(source_path)
            avg_color = img.mean(axis=(0, 1))
            txt_color = (
                int(255 - avg_color[0]),
                int(255 - avg_color[1]),
                int(255 - avg_color[2]),
            )
            for i, line in enumerate(comments.splitlines()):
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
                cv2.destroyAllWindows()

            return img
        else:
            return None

    def _plot_object_detection(
        self,
        source_path: Union[str, cv2.typing.MatLike, Image.Image],
        save_path: str = None,
        **kwargs,
    ):
        assert self.box_cls.shape[1] == 6 + self.post_cfg.get(
            "n_extra", 0
        ), f"Got unexpected shape for object detection box_cls={self.box_cls.shape}."

        img = self._read_image(source_path)

        self.labels = self.box_cls[:, 5].to(torch.int64)
        self.scores = self.box_cls[:, 4]
        self.boxes = scale_boxes(
            self.pre_cfg["YoloPre"]["img_size"],
            self.box_cls[:, :4],
            img.shape[:2],
        )
        contours = {i: [] for i in list(range(get_coco_class_num()))}

        for box, score, label in zip(self.boxes, self.scores, self.labels):
            img = cv2.putText(
                img,
                f"{get_coco_label(label)} {int(100*score)}%",
                (int(box[0]), int(box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                get_coco_det_palette(label),
                1,
                cv2.LINE_AA,
            )
            contours[label.item()].append(
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
                    get_coco_det_palette(label),
                    LW,
                )

        if save_path is not None:
            cv2.imwrite(save_path, img)
            cv2.destroyAllWindows()
        return img

    def _plot_instance_segmentation(
        self,
        source_path: Union[str, cv2.typing.MatLike, Image.Image],
        save_path=None,
        **kwargs,
    ):
        img = self._plot_object_detection(source_path, None, **kwargs)
        masks = scale_image(self.mask.permute(1, 2, 0), img.shape[:2])
        overlay = np.zeros((masks.shape[0], masks.shape[1], 3))

        for i, label in enumerate(self.labels):
            overlay = np.maximum(
                overlay,
                masks[:, :, i][:, :, np.newaxis]
                * np.array(get_coco_det_palette(label)).reshape(1, 1, 3),
            )

        total_mask = overlay.max(axis=2, keepdims=True)
        inv_mask = 1 - ALPHA * total_mask / 255
        img = (img * inv_mask + overlay * ALPHA).astype(np.uint8)

        if save_path is not None:
            cv2.imwrite(save_path, img)
            cv2.destroyAllWindows()
        return img

    def _plot_pose_estimation(
        self,
        source_path: Union[str, cv2.typing.MatLike, Image.Image],
        save_path=None,
        **kwargs,
    ):
        img = self._plot_object_detection(source_path, None, **kwargs)
        self.kpts = scale_coords(
            self.pre_cfg["YoloPre"]["img_size"],
            self.box_cls[:, 6:].reshape(-1, 17, 3),
            img.shape[:2],
        )
        for kpt in self.kpts:
            for i, (x, y, v) in enumerate(kpt):
                color_k = KEYPOINT_PALLETE[i]
                if v < self.conf_thres:
                    continue
                cv2.circle(
                    img,
                    (int(x), int(y)),
                    RADIUS,
                    color_k,
                    -1,
                    lineType=cv2.LINE_AA,
                )

            for j, sk in enumerate(POSE_SKELETON):
                pos1 = (int(kpt[sk[0] - 1, 0]), int(kpt[sk[0] - 1, 1]))
                pos2 = (int(kpt[sk[1] - 1, 0]), int(kpt[sk[1] - 1, 1]))

                conf1 = kpt[sk[0] - 1, 2]
                conf2 = kpt[sk[1] - 1, 2]

                if conf1 < self.conf_thres or conf2 < self.conf_thres:
                    continue
                cv2.line(
                    img,
                    pos1,
                    pos2,
                    LIMB_PALLETE[j],
                    thickness=int(np.ceil(LW / 2)),
                    lineType=cv2.LINE_AA,
                )
        if save_path is not None:
            cv2.imwrite(save_path, img)
            cv2.destroyAllWindows()
        return img
