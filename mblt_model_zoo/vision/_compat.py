"""Compatibility helpers for YAML-backed vision model exports.

This module rebuilds the legacy task package exports that were removed during
the YAML migration. The generated classes keep the familiar import paths and
constructor shape while delegating model loading to ``MBLT_Engine``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeAlias

from .wrapper import CoreMode, MBLT_Engine

_MODEL_DIR = Path(__file__).parent / "models"


class CompatMBLTEngine(MBLT_Engine):
    """Typed base for dynamically generated legacy compatibility wrappers."""

    def __init__(
        self,
        local_path: str | None = None,
        model_type: str = "DEFAULT",
        infer_mode: CoreMode = "global8",
        product: str = "aries",
        dev_no: int = 0,
        target_cores: Sequence[str] | None = None,
        target_clusters: Sequence[int] | None = None,
        model_path: str | None = None,
        mxq_path: str | None = None,
        onnx_path: str | None = None,
        framework: str | None = None,
    ) -> None:
        """Type-only constructor matching the generated legacy wrappers."""
        del (
            local_path,
            model_type,
            infer_mode,
            product,
            dev_no,
            target_cores,
            target_clusters,
            model_path,
            mxq_path,
            onnx_path,
            framework,
        )
        raise NotImplementedError


CompatMBLTEngineClass: TypeAlias = type[CompatMBLTEngine]


def _normalize_name(value: str) -> str:
    """Returns an alphanumeric-only, case-insensitive identifier."""

    return "".join(char for char in value.lower() if char.isalnum())


def _resolve_yaml_name(class_name: str) -> str:
    """Resolves the YAML config stem associated with a legacy class name.

    Args:
        class_name: Exported legacy class name.

    Returns:
        The YAML filename stem without the ``.yaml`` suffix.

    Raises:
        ValueError: If no unique matching YAML file can be determined.
    """

    yaml_stems = [path.stem for path in _MODEL_DIR.glob("*.yaml")]

    exact_matches = [stem for stem in yaml_stems if stem.lower() == class_name.lower()]
    if len(exact_matches) == 1:
        return exact_matches[0]

    normalized_name = _normalize_name(class_name)
    normalized_matches = [stem for stem in yaml_stems if _normalize_name(stem) == normalized_name]
    if len(normalized_matches) == 1:
        return normalized_matches[0]

    if not normalized_matches:
        raise ValueError(f"Could not find a YAML config for legacy class '{class_name}'.")

    raise ValueError(f"Found multiple YAML configs for legacy class '{class_name}': {sorted(normalized_matches)}.")


def _build_init(yaml_name: str) -> Callable[..., None]:
    """Builds a legacy-compatible ``__init__`` implementation."""

    def __init__(
        self,
        local_path: str | None = None,
        model_type: str = "DEFAULT",
        infer_mode: CoreMode = "global8",
        product: str = "aries",
        dev_no: int = 0,
        target_cores: Sequence[str] | None = None,
        target_clusters: Sequence[int] | None = None,
        model_path: str | None = None,
        mxq_path: str | None = None,
        onnx_path: str | None = None,
        framework: str | None = None,
    ) -> None:
        """Initializes a YAML-backed compatibility wrapper.

        Args:
            local_path: Deprecated legacy MXQ path alias. Prefer ``model_path``
                for generic MXQ or ONNX loading, or ``mxq_path`` for an
                explicit MXQ-only override.
            model_type: YAML config variant to load.
            infer_mode: Execution mode forwarded to ``MBLT_Engine``.
            product: Retained for backward compatibility but ignored by the
                YAML-backed registry in `2.0.0`.
            dev_no: Accelerator device number.
            target_cores: Optional core selection for single-core mode.
            target_clusters: Optional cluster selection for multi/global modes.
            model_path: Optional explicit local model path for MXQ or ONNX.
            mxq_path: Optional explicit MXQ path alias.
            onnx_path: Optional explicit ONNX path.
            framework: Execution framework, either ``"mxq"`` or ``"onnx"``. When
                omitted, ``model_path`` suffix is used first, then MXQ is the fallback.
        """

        del product
        MBLT_Engine.__init__(
            self,
            model_cls=yaml_name,
            model_type=model_type,
            model_path=model_path or "",
            mxq_path=mxq_path or local_path or "",
            onnx_path=onnx_path or "",
            dev_no=dev_no,
            core_mode=infer_mode,
            target_cores=list(target_cores) if target_cores is not None else None,
            target_clusters=list(target_clusters) if target_clusters is not None else None,
            framework=framework,
        )

    return __init__


def _build_missing_init(class_name: str, reason: str) -> Callable[..., None]:
    """Builds an ``__init__`` that fails with a clear compatibility message."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Raises an informative error for removed YAML-backed models."""

        del args, kwargs
        raise ValueError(f"Legacy vision model '{class_name}' is not available in the YAML model registry: {reason}")

    return __init__


def create_model_class(class_name: str, module_name: str) -> CompatMBLTEngineClass:
    """Creates a legacy model class that delegates to ``MBLT_Engine``.

    Args:
        class_name: Class name to expose from the task module.
        module_name: Module path that should own the generated class.

    Returns:
        A dynamically generated ``MBLT_Engine`` subclass.
    """

    class_doc = f"Compatibility wrapper for the legacy ``{class_name}`` vision model."
    try:
        yaml_name = _resolve_yaml_name(class_name)
    except ValueError as exc:
        return type(
            class_name,
            (CompatMBLTEngine,),
            {
                "__doc__": class_doc,
                "__init__": _build_missing_init(class_name, str(exc)),
                "__module__": module_name,
                "_yaml_missing": True,
            },
        )

    return type(
        class_name,
        (CompatMBLTEngine,),
        {
            "__doc__": class_doc,
            "__init__": _build_init(yaml_name),
            "__module__": module_name,
            "_yaml_name": yaml_name,
        },
    )


def export_model_classes(namespace: dict[str, Any], class_names: Iterable[str], module_name: str) -> None:
    """Populates a module namespace with generated model classes.

    Args:
        namespace: Target module globals.
        class_names: Legacy class names to generate.
        module_name: Import path for generated classes.
    """

    for class_name in class_names:
        namespace[class_name] = create_model_class(class_name, module_name)
