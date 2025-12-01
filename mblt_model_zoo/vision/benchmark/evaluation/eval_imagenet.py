import math
from time import time

from tqdm import tqdm

from ..dataloader import CustomImageFolder, get_imagenet_loader


def eval_imagenet(model, data_path, batch_size):

    dataset = CustomImageFolder(data_path)
    dataloader = get_imagenet_loader(dataset, batch_size, model.preprocess)

    num_data = len(dataset)
    total_iter = math.ceil(num_data / batch_size)
    pbar = tqdm(dataloader, total=total_iter)

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
            f"Top 1 Acc.: {acc:.3f}, FPS: {cum_num_data / inference_time:.3f}"
        )

    print(f"Top 1 Acc.: {acc:.5f}, FPS: {cum_num_data / inference_time:.3f}")
