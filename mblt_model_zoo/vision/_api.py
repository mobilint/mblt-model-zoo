"""Public helpers for discovering available vision tasks and models."""

from __future__ import annotations

import importlib
import inspect
from typing import Iterable

from ._tasks import VISION_TASKS, normalize_vision_task
from .wrapper import MBLT_Engine


def list_tasks() -> list[str]:
    """Lists the available vision tasks."""

    return list(VISION_TASKS)


def list_models(tasks: str | Iterable[str] | None = None) -> dict[str, list[str]]:
    """Lists available models for the selected vision tasks.

    Args:
        tasks: Task name or names to inspect. When omitted, all tasks are used.

    Returns:
        A mapping of task name to exported model class names.

    Raises:
        ValueError: If an unknown task name is provided.
    """

    if tasks is None:
        task_list = list(VISION_TASKS)
    elif isinstance(tasks, str):
        task_list = [tasks]
    else:
        task_list = list(tasks)

    available_models: dict[str, list[str]] = {}
    for task in task_list:
        module_name = normalize_vision_task(task)
        module = importlib.import_module(f".{module_name}", package=__name__.replace("._api", ""))
        available_models[task] = sorted(
            name
            for name, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, MBLT_Engine) and obj is not MBLT_Engine and not getattr(obj, "_yaml_missing", False)
        )

    return available_models
