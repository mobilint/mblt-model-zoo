"""Public helpers for discovering available vision tasks and models."""

from __future__ import annotations

import importlib
import inspect
from typing import Dict, Iterable, List, Union

from .wrapper import MBLT_Engine

TASKS = [
    "image_classification",
    "object_detection",
    "instance_segmentation",
    "pose_estimation",
    "face_detection",
]


def list_tasks() -> List[str]:
    """Lists the available vision tasks."""

    return TASKS.copy()


def list_models(tasks: Union[str, Iterable[str], None] = None) -> Dict[str, List[str]]:
    """Lists available models for the selected vision tasks.

    Args:
        tasks: Task name or names to inspect. When omitted, all tasks are used.

    Returns:
        A mapping of task name to exported model class names.

    Raises:
        ValueError: If an unknown task name is provided.
    """

    if tasks is None:
        task_list = TASKS
    elif isinstance(tasks, str):
        task_list = [tasks]
    else:
        task_list = list(tasks)

    invalid_tasks = sorted(set(task_list) - set(TASKS))
    if invalid_tasks:
        raise ValueError(f"mblt_model_zoo.vision supports tasks in {TASKS}, got {invalid_tasks}.")

    available_models: Dict[str, List[str]] = {}
    for task in task_list:
        module = importlib.import_module(f".{task}", package=__name__.replace("._api", ""))
        available_models[task] = sorted(
            name
            for name, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, MBLT_Engine) and obj is not MBLT_Engine and not getattr(obj, "_yaml_missing", False)
        )

    return available_models
