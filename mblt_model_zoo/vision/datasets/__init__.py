"""YAML-backed definitions for vision validation datasets."""

from .registry import get_dataset_category_ids, get_dataset_class_names, get_dataset_config, get_dataset_config_for_task

__all__ = [
    "get_dataset_category_ids",
    "get_dataset_class_names",
    "get_dataset_config",
    "get_dataset_config_for_task",
]
