"""Deprecated task-normalization compatibility exports."""

from collections.abc import Iterable

from mblt_vision._tasks import VISION_TASKS
from mblt_vision._tasks import normalize_vision_task as _normalize_task

VISION_TASK_ALIASES: dict[str, str] = {"oriented_bounding_boxes": "obb"}


def normalize_vision_task(task: str, *, supported: Iterable[str] | None = None) -> str:
    """Normalize a legacy task alias before delegating to standalone Vision."""

    normalized = VISION_TASK_ALIASES.get(task.lower(), task) if isinstance(task, str) else task
    return _normalize_task(normalized, supported=supported)


__all__ = ["VISION_TASKS", "VISION_TASK_ALIASES", "normalize_vision_task"]
