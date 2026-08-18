"""Deprecated compatibility exports for standalone Vision datasets."""

from mblt_vision.datasets import (
    get_dataset_category_ids,
    get_dataset_class_names,
    get_dataset_config,
    get_dataset_config_for_task,
)

__all__ = [
    "get_dataset_category_ids",
    "get_dataset_class_names",
    "get_dataset_config",
    "get_dataset_config_for_task",
]
