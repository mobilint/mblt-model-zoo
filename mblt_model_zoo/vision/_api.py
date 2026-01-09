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


def list_tasks():
    """
    List the supported vision tasks.

    Returns:
        List[str]: A list of supported task names.
    """
    return TASKS


def list_models(tasks: Union[str, List[str]] = None):
    """
    List available models for the specified tasks.

    Args:
        tasks (Union[str, List[str]], optional): The task or list of tasks to query.
            Defaults to all TASKS.

    Returns:
        Dict[str, List[str]]: A dictionary mapping task names to lists of available model names.
    """
    if tasks is None:
        tasks = TASKS

    if isinstance(tasks, str):
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
                inspect.isfunction(obj)
                and inspect.signature(obj).return_annotation == MBLT_Engine
            ):
                available_models[task].append(name)

    return available_models
