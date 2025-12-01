import os
from typing import Callable

import cv2
import numpy as np
import torch
from faster_coco_eval import COCO
from PIL import Image


class CustomCocodata:
    """Custom COCO dataset class (without using torchvision)"""

    def __init__(self, root, annFile):
        """
        Args:
            root (str): Image directory path
            annFile (str): Annotation file path
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int):
        """Load image by ID"""
        image_path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
        image = cv2.imread(image_path)  # Load image (BGR format)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    def __getitem__(self, index: int):
        """Get the image and target by index"""
        image_id = self.ids[index]
        image = self._load_image(image_id)
        height = self.coco.imgs[image_id]["height"]
        width = self.coco.imgs[image_id]["width"]
        return image, index, height, width

    def __len__(self):
        """Return the total number of images"""
        return len(self.ids)


def get_coco_loader(dataset, batch_size, preprocess_fn: Callable):

    def loader(batch):
        batch = filter(lambda x: x is not None, batch)
        images, idx, height, width = zip(*batch)

        processed_images = []
        for img in images:
            img = preprocess_fn(img)
            processed_images.append(img)

        height = np.array(height)
        width = np.array(width)

        return (
            np.stack(processed_images, axis=0),
            np.stack((height, width), axis=1),
            idx,
        )

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=loader
    )


class CustomImageFolder:
    def __init__(self, root):
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(root)
        self.make_dataset()

    def make_dataset(self):
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

    def loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def find_classes(self, directory):
        classes = sorted([d.name for d in os.scandir(directory) if d.is_dir()])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target

    def __len__(self):
        return len(self.samples)


def get_imagenet_loader(dataset, batch_size, preprocess_fn: Callable):

    def loader(batch):
        batch = filter(lambda x: x is not None, batch)  # remove None
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
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=loader
    )


class CustomWiderface:
    def __init__(self, root):
        """
        Args:
            root (str): Image directory path
        """
        self.root = root
        self.classes = self.find_classes(root)
        self.make_dataset()

    def make_dataset(self):
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

    def loader(self, image_path):
        """Load image by image path"""
        image = cv2.imread(image_path)  # Load image (BGR format)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    def find_classes(self, directory):
        unsorted_classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        class_to_idx = {}
        for cls_name in unsorted_classes:
            cls_idx = int(cls_name.split("--")[0])
            class_to_idx[cls_name] = cls_idx

        sorted_classes = sorted(
            class_to_idx.keys(), key=lambda x: class_to_idx[x]
        )  # sort by dictionary value with ascending order

        return sorted_classes

    def __getitem__(self, index: int):
        """Get the image and target by index"""
        image_path, target_class, fname = self.samples[index]
        image = self.loader(image_path)

        return image, target_class, fname

    def __len__(self):
        """Return the total number of images"""
        return len(self.samples)


def get_widerface_loader(dataset, batch_size, preprocess_fn: Callable):
    def loader(batch):
        batch = filter(lambda x: x is not None, batch)
        images, target_classes, fnames = zip(*batch)
        processed_images = []
        heights = []
        widths = []
        for img in images:
            height, width = img.shape[:2]
            processed_images.append(preprocess_fn(img))
            heights.append(height)
            widths.append(width)

        return (
            np.stack(processed_images, axis=0),
            np.stack((heights, widths), axis=1),
            target_classes,
            fnames,
        )

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=loader
    )
