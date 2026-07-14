"""
Evaluation script for ImageNet dataset.
"""

from __future__ import annotations

import math
from time import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from ..datasets import CustomImageFolder, get_imagenet_loader

if TYPE_CHECKING:
    from ...wrapper import MBLT_Engine


def eval_imagenet(model: MBLT_Engine, data_path: str, batch_size: int) -> float:
    """Evaluates a classification model on the ImageNet validation set.

    Computes Top-1 and Top-5 accuracy and inference speed (FPS) on the NPU.

    Args:
        model (MBLT_Engine): The vision engine to evaluate.
        data_path (str): Path to the ImageNet validation images.
        batch_size (int): Number of images per inference batch.

    Returns:
        float: Calculated Top-1 accuracy (range 0.0 to 1.0).
    """
    dataset = CustomImageFolder(data_path)
    dataloader = get_imagenet_loader(dataset, batch_size, model.preprocess)
    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter, desc="Evaluating ImageNet")
    inference_time = 0.0
    infer_post_time = 0.0
    total_time = 0.0
    cum_num_data = 0
    cum_top1_correct = 0
    cum_top5_correct = 0
    top1_acc = 0.0
    top5_acc = 0.0
    for input_npu, label in pbar:
        cum_num_data += len(label)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic
        result = model.postprocess(out_npu)
        infer_post_time += time() - tic
        output = result.output
        if isinstance(output, np.ndarray):
            prediction = output.argmax(-1)
        elif isinstance(output, torch.Tensor):
            prediction = output.argmax(-1).cpu().numpy()
        else:
            raise TypeError(f"Expected classification output to be a tensor or ndarray, got {type(output)}.")
        top_k = min(5, output.shape[-1])
        if isinstance(output, torch.Tensor):
            top5_prediction = output.topk(top_k, dim=-1).indices.cpu().numpy()
        else:
            top5_prediction = np.argpartition(output, -top_k, axis=-1)[:, -top_k:]
        label_array = np.asarray(label)
        cum_top1_correct += (prediction == label_array).sum().item()
        cum_top5_correct += np.any(top5_prediction == label_array[:, np.newaxis], axis=-1).sum().item()
        top1_acc = cum_top1_correct / cum_num_data
        top5_acc = cum_top5_correct / cum_num_data
        total_time += time() - tic
        pbar.set_postfix_str(
            f"Top 1 Acc.: {100 * top1_acc:.3f}%, Top 5 Acc.: {100 * top5_acc:.3f}%, "
            f"NPU FPS: {cum_num_data / inference_time:.3f}"
        )
    pbar.close()
    print("ImageNet evaluation completed")
    print(
        f"Top 1 Acc.: {100 * top1_acc:.3f}%, "
        f"Top 5 Acc.: {100 * top5_acc:.3f}%, "
        f"NPU FPS: {cum_num_data / inference_time:.3f}"
    )
    return top1_acc
