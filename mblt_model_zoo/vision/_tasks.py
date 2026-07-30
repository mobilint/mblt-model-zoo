"""Canonical task handling shared by Vision APIs and tooling."""

from __future__ import annotations

from collections.abc import Iterable

VISION_TASKS: tuple[str, ...] = (
    "image_classification",
    "depth_estimation",
    "object_detection",
    "instance_segmentation",
    "semantic_segmentation",
    "obb",
    "pose_estimation",
    "face_detection",
)

VISION_TASK_ALIASES: dict[str, str] = {
    "oriented_bounding_boxes": "obb",
}


def normalize_vision_task(task: str, *, supported: Iterable[str] | None = None) -> str:
    """Return the canonical lowercase name for a supported Vision task.

    Args:
        task: Task name to normalize.
        supported: Optional subset of canonical tasks accepted by the caller.

    Returns:
        Canonical Vision task name.

    Raises:
        TypeError: If ``task`` is not a string.
        ValueError: If the normalized task is unsupported.
    """

    if not isinstance(task, str):
        raise TypeError(f"Vision task must be a string, got {type(task).__name__}.")
    normalized = VISION_TASK_ALIASES.get(task.lower(), task.lower())
    supported_tasks = tuple(VISION_TASKS if supported is None else supported)
    if normalized not in supported_tasks:
        aliases = {alias for alias, value in VISION_TASK_ALIASES.items() if value in supported_tasks}
        accepted = sorted(set(supported_tasks) | aliases)
        raise ValueError(f"Unsupported Vision task {task!r}; expected one of {accepted}.")
    return normalized
