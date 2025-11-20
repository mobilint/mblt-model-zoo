import importlib
import inspect
from typing import Dict, List, Union

from .types import TransformersModelInfo

TASKS = [
    "large_language_model",
    "speech_to_text",
    "text_to_speech",
    "vision_language_model",
]


def list_tasks():
    return TASKS


def list_models(
    tasks: Union[str, List[str]] = TASKS,
) -> Dict[str, List[TransformersModelInfo]]:
    if isinstance(tasks, str):
        tasks = [tasks]
    assert set(tasks).issubset(TASKS), f"mblt model zoo supports tasks in {TASKS}"

    available_models: Dict[str, List[TransformersModelInfo]] = {}
    for task in tasks:
        available_models[task] = []
        try:
            module = importlib.import_module(
                f"..{task}", package=__name__.replace("._api", "")
            )
        except ImportError as e:
            print(f"Failed to import module for task '{task}': {e}")
            continue

        for name, obj in inspect.getmembers(module):
            if (
                not inspect.isclass(obj)
                and isinstance(obj, TransformersModelInfo)
                and obj is not TransformersModelInfo
            ):
                available_models[task].append(obj)

    return available_models
