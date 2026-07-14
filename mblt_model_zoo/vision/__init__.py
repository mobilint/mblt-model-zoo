"""MBLT vision task exports and discovery helpers.

The vision package keeps task subpackages as the preferred import surface while
also supporting legacy top-level model imports such as
``from mblt_model_zoo.vision import ResNet50``.
"""

from __future__ import annotations

from . import face_detection as face_detection
from . import image_classification as image_classification
from . import instance_segmentation as instance_segmentation
from . import object_detection as object_detection
from . import oriented_bounding_boxes as oriented_bounding_boxes
from . import pose_estimation as pose_estimation
from ._api import list_models as list_models
from ._api import list_tasks as list_tasks
from .wrapper import MBLT_Engine as MBLT_Engine

_TASK_MODULES = (
    face_detection,
    image_classification,
    instance_segmentation,
    object_detection,
    oriented_bounding_boxes,
    pose_estimation,
)

_LEGACY_MODEL_EXPORTS: dict[str, object] = {}
for _task_module in _TASK_MODULES:
    for _export_name in getattr(_task_module, "__all__", ()):
        if _export_name in _LEGACY_MODEL_EXPORTS:
            raise RuntimeError(f"Duplicate vision export detected for '{_export_name}'.")
        _LEGACY_MODEL_EXPORTS[_export_name] = _task_module

_PUBLIC_EXPORTS = [
    "MBLT_Engine",
    "list_models",
    "list_tasks",
    "face_detection",
    "image_classification",
    "instance_segmentation",
    "object_detection",
    "oriented_bounding_boxes",
    "pose_estimation",
] + sorted(_LEGACY_MODEL_EXPORTS)
# Keep legacy compatibility exports synchronized with their task packages.
__all__: list[str] = _PUBLIC_EXPORTS  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> object:
    """Lazily resolve legacy top-level model exports.

    Args:
        name: Attribute requested from the vision package.

    Returns:
        The exported model wrapper class for the requested legacy name.

    Raises:
        AttributeError: If the requested name is not exported by the package.
    """

    task_module = _LEGACY_MODEL_EXPORTS.get(name)
    if task_module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(task_module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package attributes including lazy legacy exports."""

    return sorted(set(globals()) | set(__all__))
