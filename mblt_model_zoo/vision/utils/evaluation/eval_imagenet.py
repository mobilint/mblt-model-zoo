import math
from time import time

from tqdm import tqdm

from ...wrapper import MBLT_Engine
from ..datasets import load_dataloader


def eval_imagenet(
    model: MBLT_Engine,
    imagenet_eval_cfg: dict,
    batch_size: int = 1,
):
    """Evaluate the model on the imagenet dataset

    Args:
        model (Model Class): Model class object
        helper (Helper Class): Helper class object with pre_process and post_process methods
        imagenet_eval_cfg (dict): Configuration dictionary for the evaluation

    Returns:
        acc, inference_time, fps: Accuracy, Inference time, and Frames per second
    """
    dataset, dataloader = load_dataloader(
        dataset_name="imagenet",
        dataset_path=imagenet_eval_cfg["dataset_path"],
        preprocess=model.preprocess,
        batch_size=batch_size,
    )

    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter)

    inference_time = 0
    infer_post_time = 0
    total_time = 0

    cum_num_data = 0
    cum_correct = 0
    acc = 0

    pbar = tqdm(dataloader, total=total_iter)

    for input_npu, label in pbar:
        cum_num_data += len(label)
        tic = time()
        out_npu = model(input_npu)
        inference_time += time() - tic

        logit = model.postprocess(out_npu)
        infer_post_time += time() - tic
        cum_correct += (logit.argmax(-1).cpu().numpy() == label).sum().item()
        acc = cum_correct / cum_num_data
        total_time += time() - tic
        pbar.set_postfix_str(
            f"Top 1 Acc.: {acc:.3f}, FPS: {cum_num_data / inference_time:.3f}"
        )

    return (
        acc,
        inference_time,
        cum_num_data / inference_time,
        infer_post_time,
        cum_num_data / total_time,
        total_time,
        cum_num_data / total_time,
    )
