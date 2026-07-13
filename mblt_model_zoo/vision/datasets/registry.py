"""Load YAML-backed vision dataset definitions."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DATASET_CONFIG_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def _load_dataset_config(name: str) -> dict[str, Any]:
    """Load and validate a dataset definition without mutating its cached value."""

    config_path = DATASET_CONFIG_DIR / f"{name.lower()}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Vision dataset definition not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Vision dataset definition must be a mapping: {config_path}")
    if not isinstance(config.get("path"), str):
        raise ValueError(f"Vision dataset definition requires a string `path`: {config_path}")
    if not isinstance(config.get("name"), str):
        raise ValueError(f"Vision dataset definition requires a string `name`: {config_path}")
    if not isinstance(config.get("tasks"), list) or not all(isinstance(task, str) for task in config["tasks"]):
        raise ValueError(f"Vision dataset definition requires a string-list `tasks`: {config_path}")
    config["path"] = str(Path(config["path"]).expanduser())
    return config


def get_dataset_config(name: str) -> dict[str, Any]:
    """Load a named vision dataset definition.

    Args:
        name: Dataset filename stem, such as ``dotav1``.

    Returns:
        Parsed dataset configuration with its path expanded.

    Raises:
        FileNotFoundError: If the dataset definition does not exist.
        ValueError: If the definition is malformed.
    """

    return copy.deepcopy(_load_dataset_config(name))


def get_dataset_class_names(name: str) -> tuple[str, ...]:
    """Return ordered class names from a dataset definition.

    Args:
        name: Dataset filename stem, such as ``coco``.

    Returns:
        Class names ordered by contiguous zero-based index.

    Raises:
        ValueError: If the dataset has no valid contiguous ``names`` mapping.
    """

    names = _load_dataset_config(name).get("names")
    if not isinstance(names, dict) or not all(
        isinstance(index, int) and isinstance(label, str) for index, label in names.items()
    ):
        raise ValueError(f"Vision dataset definition requires an integer-keyed `names` mapping: {name}")
    if set(names) != set(range(len(names))):
        raise ValueError(f"Vision dataset class IDs must be contiguous and zero-based: {name}")
    return tuple(names[index] for index in range(len(names)))


def get_dataset_config_for_task(task: str) -> dict[str, Any]:
    """Return the validation dataset definition associated with a vision task.

    Args:
        task: Vision task name from a model postprocess configuration.

    Returns:
        Matching dataset configuration.

    Raises:
        ValueError: If no configured dataset supports the task.
    """

    for config_path in sorted(DATASET_CONFIG_DIR.glob("*.yaml")):
        config = get_dataset_config(config_path.stem)
        if task in config["tasks"]:
            return config
    raise ValueError(f"No vision dataset definition supports task `{task}`.")
