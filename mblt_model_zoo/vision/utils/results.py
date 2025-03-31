import numpy as np
import torch
from typing import Union, List
import os
import cv2
from .datasets import *


class Results:
    def __init__(self, pre_cfg: dict, post_cfg: dict, output):
        self.pre_cfg = pre_cfg
        self.post_cfg = post_cfg
        self.task = post_cfg["task"]
        self.output = output

    @classmethod
    def from_engine(cls, engine, output):
        pre_cfg = engine.pre_cfg
        post_cfg = engine.post_cfg
        return cls(pre_cfg, post_cfg, output)

    def plot(self, source_path, save_path=None, **kwargs):
        if self.task.lower() == "image_classification":
            return self._plot_image_classification(source_path, save_path, **kwargs)
        elif self.task.lower() == "object_detection":
            return self._plot_object_detection(source_path, save_path, **kwargs)
        elif self.task.lower() == "image_segmentation":
            return self._plot_image_segmentation(source_path, save_path, **kwargs)
        elif self.task.lower() == "image_segmentation_instance":
            return self._plot_image_segmentation_instance(
                source_path, save_path, **kwargs
            )
        else:
            raise NotImplementedError(
                f"Task {self.task} is not supported for plotting results."
            )

    def _plot_image_classification(
        self, source_path, save_path=None, topk=5, mode="draw", **kwargs
    ):
        assert mode in [
            "draw",
            "print",
        ], f"Mode {mode} is not supported. Available modes: ['draw', 'print']"

        if mode == "print":
            # print the topk results with corresponding labels and probabilities
            # assume self.output is a probability vector
            if isinstance(self.output, np.ndarray):
                self.output = torch.tensor(self.output)

            topk_probs, topk_indices = torch.topk(self.output, topk)
            topk_probs = topk_probs.squeeze().numpy()
            topk_indices = topk_indices.squeeze().numpy()

            # load labels
            labels = [get_imagenet_label(i) for i in topk_indices]
            for i in range(topk):
                print(f"Label: {labels[i]}, Probability: {topk_probs[i]*100:.2f}%")
        else:
            raise NotImplementedError(
                f"Mode {mode} is not supported for plotting results."
            )

    def _plot_object_detection(self, source_path, save_path=None, **kwargs):
        pass

    def _plot_image_segmentation(self, source_path, save_path=None, **kwargs):
        pass

    def _plot_image_segmentation_instance(self, source_path, save_path=None, **kwargs):
        pass
