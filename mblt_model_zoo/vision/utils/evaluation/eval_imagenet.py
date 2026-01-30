"""
Evaluation script for ImageNet dataset.
"""

import math
from time import time

from tqdm import tqdm

from ..datasets import CustomImageFolder, get_imagenet_loader


def eval_imagenet(model, data_path, batch_size):
    """
    Evaluate a model on the ImageNet dataset.
    Args:
        model: The model to evaluate.
        data_path (str): Path to the ImageNet data.
        batch_size (int): Batch size for evaluation.
    Returns:
        float: Top-1 accuracy of the model.
    """

    dataset = CustomImageFolder(data_path)
    dataloader = get_imagenet_loader(dataset, batch_size, model.preprocess)

    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter, desc="Evaluating ImageNet")

    inference_time = 0
    infer_post_time = 0
    total_time = 0

    cum_num_data = 0
    cum_correct = 0
    acc = 0

    for input_npu, label in pbar:
        cum_num_data += len(label)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic
        result = model.postprocess(out_npu)
        infer_post_time += time() - tic
        cum_correct += (result.output.argmax(-1).cpu().numpy() == label).sum().item()
        acc = cum_correct / cum_num_data
        total_time += time() - tic
        pbar.set_postfix_str(
            f"Top 1 Acc.: {100*acc:.3f}%, NPU FPS: {cum_num_data / inference_time:.3f}"
        )
    pbar.close()
    print("ImageNet evaluation completed")
    print(f"Top 1 Acc.: {100*acc:.3f}%, NPU FPS: {cum_num_data / inference_time:.3f}")
    return acc
