"""
Custom dataloaders for vision datasets.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import cv2
import numpy as np
import torch
from faster_coco_eval import COCO
from PIL import Image


class CustomCocodata(torch.utils.data.Dataset[tuple[np.ndarray, int, int, int]]):
    """Custom COCO dataset class for loading images and metadata.

    This class provides a simple interface for accessing COCO formatted data
    without requiring external library dependencies like torchvision.

    Attributes:
        root (str): Root directory path containing the images.
        coco (COCO): COCO helper object from faster_coco_eval.
        ids (list[int]): Sorted list of image IDs in the dataset.
    """

    def __init__(self, root: str, annFile: str, min_keypoints: int | None = None) -> None:
        """Initializes the CustomCocodata instance.

        Args:
            root (str): Path to the directory containing images.
            annFile (str): Path to the COCO annotation JSON file.
            min_keypoints: If set, keep only images with at least one
                annotation whose ``num_keypoints`` is greater than this value.
        """
        self.root = root
        self.coco = COCO(annFile)
        if min_keypoints is None:
            self.ids = list(sorted(self.coco.imgs.keys()))
        else:
            self.ids = list(
                sorted(
                    {ann["image_id"] for ann in self.coco.anns.values() if ann.get("num_keypoints", 0) > min_keypoints}
                )
            )

    def _load_image(self, image_id: int) -> np.ndarray:
        """Load image by ID"""
        image_path = os.path.join(self.root, self.coco.loadImgs(image_id)[0]["file_name"])
        image = cv2.imread(image_path)  # Load image (BGR format)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, int, int]:
        """Get the image and target by index"""
        image_id = self.ids[index]
        image = self._load_image(image_id)
        height = self.coco.imgs[image_id]["height"]
        width = self.coco.imgs[image_id]["width"]
        return image, index, height, width

    def __len__(self) -> int:
        """Return the total number of images"""
        return len(self.ids)


def get_coco_loader(dataset: CustomCocodata, batch_size: int, preprocess_fn: Callable) -> torch.utils.data.DataLoader:
    """Creates a DataLoader for the COCO dataset.

    Args:
        dataset (CustomCocodata): The dataset instance to load from.
        batch_size (int): Number of samples per batch.
        preprocess_fn (Callable): Function used to preprocess images.

    Returns:
        torch.utils.data.DataLoader: A configured DataLoader for the COCO dataset.
    """

    def loader(batch: list[Any]) -> tuple[np.ndarray, np.ndarray, list[Any], tuple[int, ...]]:
        """Collate function for COCO DataLoader."""
        batch = list(filter(lambda x: x is not None, batch))
        images, idx, height, width = zip(*batch)

        processed_images = []
        ratio_pads = []
        for img in images:
            processed = preprocess_fn(img)
            if isinstance(processed, tuple) and len(processed) == 2 and isinstance(processed[1], dict):
                processed_img, metadata = processed
                ratio_pads.append(metadata.get("ratio_pad"))
            else:
                processed_img = processed
                ratio_pads.append(None)
            processed_images.append(processed_img)

        height_arr = np.array(height)
        width_arr = np.array(width)

        return (
            np.stack(processed_images, axis=0),
            np.stack((height_arr, width_arr), axis=1),
            ratio_pads,
            idx,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=loader,
    )


class CustomDOTAv1(torch.utils.data.Dataset[tuple[np.ndarray, str, int, int]]):
    """Custom DOTAv1 validation dataset for OBB evaluation.

    Attributes:
        root: DOTAv1 dataset root.
        image_root: Directory containing validation images.
        ids: Image IDs derived from file stems.
    """

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(self, root: str) -> None:
        """Initializes the DOTAv1 validation dataset.

        Args:
            root: DOTAv1 root containing ``images/val`` and ``labels/val``.

        Raises:
            FileNotFoundError: If the validation image directory is missing.
        """
        self.root = root
        self.image_root = os.path.join(root, "images", "val")
        if not os.path.isdir(self.image_root):
            raise FileNotFoundError(f"DOTAv1 image directory not found: {self.image_root}")
        self.image_paths = [
            os.path.join(self.image_root, file_name)
            for file_name in sorted(os.listdir(self.image_root))
            if file_name.lower().endswith(self.IMG_EXTENSIONS)
        ]
        self.ids = [os.path.splitext(os.path.basename(path))[0] for path in self.image_paths]

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load an image as RGB."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> tuple[np.ndarray, str, int, int]:
        """Get the image and metadata by index."""
        image_path = self.image_paths[index]
        image = self._load_image(image_path)
        height, width = image.shape[:2]
        return image, self.ids[index], height, width

    def __len__(self) -> int:
        """Return the number of validation images."""
        return len(self.image_paths)


def get_dota_loader(dataset: CustomDOTAv1, batch_size: int, preprocess_fn: Callable) -> torch.utils.data.DataLoader:
    """Creates a DataLoader for DOTAv1 validation.

    Args:
        dataset: The DOTAv1 dataset instance.
        batch_size: Number of samples per batch.
        preprocess_fn: Function used to preprocess images.

    Returns:
        Configured DataLoader for DOTAv1.
    """

    def loader(batch: list[Any]) -> tuple[np.ndarray, np.ndarray, list[Any], tuple[str, ...]]:
        """Collate function for DOTAv1 DataLoader."""
        batch = list(filter(lambda x: x is not None, batch))
        images, image_ids, height, width = zip(*batch)

        processed_images = []
        ratio_pads = []
        for img in images:
            processed = preprocess_fn(img)
            if isinstance(processed, tuple) and len(processed) == 2 and isinstance(processed[1], dict):
                processed_img, metadata = processed
                ratio_pads.append(metadata.get("ratio_pad"))
            else:
                processed_img = processed
                ratio_pads.append(None)
            processed_images.append(processed_img)

        return (
            np.stack(processed_images, axis=0),
            np.stack((np.array(height), np.array(width)), axis=1),
            ratio_pads,
            image_ids,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=loader,
    )


class CustomImageFolder(torch.utils.data.Dataset[tuple[Image.Image, int]]):
    """Custom ImageFolder dataset for loading images from class-based directory structures.

    Expects data to be organized in the format: root/class_name/image.jpg.

    Attributes:
        root (str): Root directory path.
        classes (list[str]): List of class names found in the root directory.
        class_to_idx (dict): Mapping from class name to class index.
        samples (list[tuple]): List of (image_path, class_index) tuples.
    """

    def __init__(self, root: str) -> None:
        """Initializes the CustomImageFolder instance.

        Args:
            root (str): Path to the root directory.
        """
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(root)
        self.samples: list[tuple[str, int]] = []
        self.make_dataset()

    def make_dataset(self) -> None:
        """Scans the root directory to create a list of samples."""
        instances = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

        self.samples = instances

    def loader(self, path: str) -> Image.Image:
        """Load image from path using PIL."""
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """Find classes in the specified directory."""
        classes = sorted([d.name for d in os.scandir(directory) if d.is_dir()])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        """
        Get sample and target at the specified index.
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            tuple: (sample, target) where sample is the loaded image and target is the class index.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target

    def __len__(self) -> int:
        """
        Return the total number of samples.
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.samples)


def get_imagenet_loader(
    dataset: CustomImageFolder, batch_size: int, preprocess_fn: Callable
) -> torch.utils.data.DataLoader:
    """Creates a DataLoader for the ImageNet dataset.

    Args:
        dataset (CustomImageFolder): The dataset instance to load from.
        batch_size (int): Number of samples per batch.
        preprocess_fn (Callable): Function used to preprocess images.

    Returns:
        torch.utils.data.DataLoader: A configured DataLoader for the ImageNet dataset.
    """

    def loader(batch: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Collate function for ImageNet DataLoader."""
        batch = list(filter(lambda x: x is not None, batch))  # remove None
        images, labels = zip(*batch)
        processed_images = []
        for img in images:
            img = preprocess_fn(img)
            processed_images.append(img)

        return (
            np.stack(processed_images, axis=0),
            np.array(labels),
        )  # BHWC, labels

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=loader,
    )


class CustomWiderface(torch.utils.data.Dataset[tuple[np.ndarray, str, str]]):
    """Custom dataset class for the WiderFace dataset.

    Attributes:
        root (str): Path to the root directory containing WiderFace images.
        classes (list[str]): List of class/event names found in the root.
        samples (list[tuple]): List of (image_path, class_name, file_name) tuples.
    """

    def __init__(self, root: str) -> None:
        """Initializes the CustomWiderface instance.

        Args:
            root (str): Path to the directory containing WiderFace images.
        """
        self.root = root
        self.classes = self.find_classes(root)
        self.samples: list[tuple[str, str, str]] = []
        self.make_dataset()

    def make_dataset(self) -> None:
        """Scans the root directory to create a list of samples."""
        instances = []
        for target_class in self.classes:
            target_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, target_class, fname
                    instances.append(item)

        self.samples = instances

    def loader(self, image_path: str) -> np.ndarray:
        """Load image by image path"""
        image = cv2.imread(image_path)  # Load image (BGR format)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    def find_classes(self, directory: str) -> list[str]:
        """Find classes in the specified directory."""
        unsorted_classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        class_to_idx = {}
        for cls_name in unsorted_classes:
            cls_idx = int(cls_name.split("--")[0])
            class_to_idx[cls_name] = cls_idx

        sorted_classes = sorted(
            class_to_idx.keys(), key=lambda x: class_to_idx[x]
        )  # sort by dictionary value with ascending order

        return sorted_classes

    def __getitem__(self, index: int) -> tuple[np.ndarray, str, str]:
        """
        Get the image and target by index.
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, target_class, fname) where image is the loaded image in RGB format.
        """
        image_path, target_class, fname = self.samples[index]
        image = self.loader(image_path)

        return image, target_class, fname

    def __len__(self) -> int:
        """
        Return the total number of images.
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.samples)


def get_widerface_loader(
    dataset: CustomWiderface, batch_size: int, preprocess_fn: Callable
) -> torch.utils.data.DataLoader:
    """Creates a DataLoader for the WiderFace dataset.

    Args:
        dataset (CustomWiderface): The dataset instance to load from.
        batch_size (int): Number of samples per batch.
        preprocess_fn (Callable): Function used to preprocess images.

    Returns:
        torch.utils.data.DataLoader: A configured DataLoader for the WiderFace dataset.
    """

    def loader(
        batch: list[Any],
    ) -> tuple[np.ndarray, np.ndarray, list[Any | None], tuple[str, ...], tuple[str, ...]]:
        """Collate function for WiderFace DataLoader."""
        batch = list(filter(lambda x: x is not None, batch))
        images, target_classes, fnames = zip(*batch)
        processed_images = []
        heights = []
        widths = []
        ratio_pads = []
        for img in images:
            height, width = img.shape[:2]
            processed = preprocess_fn(img)
            if isinstance(processed, tuple):
                processed_img, metadata = processed
                ratio_pads.append(metadata.get("ratio_pad"))
            else:
                processed_img = processed
                ratio_pads.append(None)
            processed_images.append(processed_img)
            heights.append(height)
            widths.append(width)

        return (
            np.stack(processed_images, axis=0),
            np.stack((heights, widths), axis=1),
            ratio_pads,
            target_classes,
            fnames,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=loader,
    )
