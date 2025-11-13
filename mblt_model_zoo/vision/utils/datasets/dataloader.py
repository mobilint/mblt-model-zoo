import numpy as np
import torch

from .dataset import CustomCocodata, CustomImageFolder, CustomWiderface


def load_dataloader(
    dataset_name: str,
    dataset_path: str,
    preprocess: callable,
    annotation_path: str = None,
    batch_size: int = 1,
):
    if dataset_name.lower() == "coco":
        dataset = CustomCocodata(dataset_path, annotation_path)

        def custom_collate_fn(batch):
            """
            Preprocess each image in batch to get the same size (H, W, C) and return the batch of preprocessed images (B, H, W, C)
            Also, for each preprocessed image, return the original size information (H, W) of shape (B, 2)
            """

            batch = filter(lambda x: x is not None, batch)  # remove None
            images, idx, height, width = zip(*batch)

            processed_images = []
            for img in images:
                img = preprocess(img)
                processed_images.append(img)

            height = np.array(height)
            width = np.array(width)
            return (
                np.concatenate(processed_images),
                np.stack((height, width), axis=1),
                idx,
            )  # batch (BHWC), shape_info (B, 2), idx (B)

    elif dataset_name.lower() == "imagenet":
        dataset = CustomImageFolder(dataset_path)

        def custom_collate_fn(batch):
            """
            Preprocess each image in batch to get the same size
            Keep the label as it is
            """
            batch = filter(lambda x: x is not None, batch)  # remove None
            images, labels = zip(*batch)

            processed_images = []
            for img in images:
                img = preprocess(img)
                processed_images.append(img)

            return (
                np.concatenate(processed_images, axis=0),
                np.array(labels),
            )  # BHWC, labels

    elif dataset_name.lower() == "widerface":
        dataset = CustomWiderface(dataset_path)

        def custom_collate_fn(batch):
            """
            Preprocess each image in batch to get the same size (H, W, C) and return the batch of preprocessed images (B, H, W, C)
            Also, for each preprocessed image, return the original size information (H, W) of shape (B, 2)
            """
            batch = filter(lambda x: x is not None, batch)  # remove None
            images, target_classes, fnames = zip(*batch)

            processed_images = []
            heights = []
            widths = []
            for img in images:
                height, width = img.shape[:2]
                img = preprocess(img)
                processed_images.append(img)
                heights.append(height)
                widths.append(width)
            heights = np.array(heights)
            widths = np.array(widths)

            return (
                np.concatenate(processed_images),
                np.stack((heights, widths), axis=1),
                target_classes,
                fnames,
            )

    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=custom_collate_fn,
    )

    return dataset, data_loader
