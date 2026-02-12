"""
API functions for listing tasks and models.
"""

import importlib
import inspect
from typing import List, Union

from .wrapper import MBLT_Engine

TASKS = [
    "image_classification",
    "object_detection",
    "instance_segmentation",
    "pose_estimation",
]


def list_tasks() -> List[str]:
    """Lists the available vision tasks.

    Returns:
        List[str]: A list of task names.
    """
    return TASKS


def list_models(tasks: Union[str, List[str]] = None) -> dict:
    """Lists the available models for the specified tasks.

    Args:
        tasks (Union[str, List[str]], optional): The task(s) to list models for.
            Defaults to all TASKS.

    Returns:
        dict: A dictionary where keys are task names and values are lists of model names.
    """

    if tasks is None:
        tasks = TASKS
    elif isinstance(tasks, str):
        tasks = [tasks]
    assert set(tasks).issubset(TASKS), f"mblt model zoo supports tasks in {TASKS}"

    available_models = {}
    for task in tasks:
        available_models[task] = []
        try:
            module = importlib.import_module(
                f".{task}", package=__name__.replace("._api", "")
            )
        except ImportError as e:
            print(f"Failed to import module for task '{task}': {e}")
            continue

        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, MBLT_Engine)
                and obj is not MBLT_Engine
            ):
                available_models[task].append(name)

    return available_models
