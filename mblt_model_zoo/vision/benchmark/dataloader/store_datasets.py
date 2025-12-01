import os

from datasets import load_dataset
from tqdm import tqdm


def store_imagenet(root: str = None):
    """
    Store ImageNet dataset in the given root directory

    Args:
        root (str): Root directory to store the dataset. If not provided, it will be stored in ~/.mblt_model_zoo/datasets/imagenet
    """
    if root is None:
        root = "~/.mblt_model_zoo/datasets/imagenet"
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)
    imagenet_dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        data_files={"validation": "data/validation*.parquet"},
        split="validation",
        verification_mode="no_checks",
    )

    for sample in tqdm(imagenet_dataset, desc="Storing ImageNet"):
        label = sample["label"]
        image = sample["image"]

        os.makedirs(os.path.join(root, str(label).zfill(3)), exist_ok=True)
        prev_images = len(os.listdir(os.path.join(root, str(label).zfill(3))))
        image.save(os.path.join(root, str(label).zfill(3), f"{prev_images:04d}.JPEG"))


if __name__ == "__main__":
    store_imagenet()
