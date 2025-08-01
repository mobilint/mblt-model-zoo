from typing import Union, List
import importlib
import inspect
from .wrapper import MBLT_Engine


TASKS = [
    "image_classification",
    "object_detection",
    "instance_segmentation",
    "pose_estimation",
]


def list_tasks():
    return TASKS


def list_models(tasks: Union[str, List[str]] = TASKS):
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
