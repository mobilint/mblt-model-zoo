"""Framework and local-artifact path resolution for vision engines."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_FRAMEWORKS = {"mxq", "onnx"}


def framework_from_model_path(model_path: str) -> str | None:
    """Infer the runtime framework from a local model path suffix."""

    suffix = Path(model_path).suffix.lower()
    if suffix == ".mxq":
        return "mxq"
    if suffix == ".onnx":
        return "onnx"
    return None


def uses_shifted_engine_model_path_layout(
    model_path: object,
    mxq_path: object,
    dev_no: object,
    core_mode: object,
    target_cores: object,
    postprocess_kwargs: object,
    framework: object,
    onnx_providers: object,
) -> bool:
    """Return whether engine arguments use the model-path-first layout.

    Public constructor layouts have used the third positional argument for
    either ``mxq_path`` or ``model_path``. An ONNX suffix identifies the
    model-path-first layout because it changes runtime routing. For MXQ,
    remapping is needed only when later values have the types produced by a
    one-slot positional shift; a path by itself behaves identically as the
    ``mxq_path`` alias.
    """

    if not isinstance(mxq_path, str):
        return False
    inferred_framework = framework_from_model_path(mxq_path)
    if inferred_framework not in SUPPORTED_FRAMEWORKS:
        return False
    if isinstance(model_path, str) and model_path and model_path.lower() not in SUPPORTED_FRAMEWORKS:
        return False
    if inferred_framework == "onnx":
        return True
    return (
        isinstance(dev_no, str)
        or isinstance(core_mode, int)
        or isinstance(target_cores, str)
        or (postprocess_kwargs is not None and not isinstance(postprocess_kwargs, dict))
        or isinstance(framework, dict)
        or isinstance(onnx_providers, str)
        or (model_path is not None and not isinstance(model_path, str))
        or (isinstance(model_path, str) and model_path.lower() in SUPPORTED_FRAMEWORKS)
    )


def uses_shifted_compat_model_path_layout(
    model_path: object,
    mxq_path: object,
    onnx_path: object,
    framework: object,
) -> bool:
    """Return whether generated-wrapper arguments use a shifted model-path tail."""

    if not isinstance(mxq_path, str):
        return False
    inferred_framework = framework_from_model_path(mxq_path)
    if inferred_framework not in SUPPORTED_FRAMEWORKS:
        return False
    if isinstance(model_path, str) and model_path and model_path.lower() not in SUPPORTED_FRAMEWORKS:
        return False
    if inferred_framework == "onnx":
        return True
    return (
        (isinstance(model_path, str) and model_path.lower() in SUPPORTED_FRAMEWORKS)
        or (isinstance(onnx_path, str) and framework_from_model_path(onnx_path) == "mxq")
        or (isinstance(framework, str) and framework_from_model_path(framework) is not None)
    )


def resolve_framework(framework: str | None, model_path: str = "") -> str:
    """Resolve the execution framework from explicit input and model path."""

    normalized_framework = framework.lower() if framework is not None else None
    if normalized_framework is not None and normalized_framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework: {framework}. Must be one of {sorted(SUPPORTED_FRAMEWORKS)}.")
    inferred_framework = framework_from_model_path(model_path) if model_path else None
    if normalized_framework and inferred_framework and normalized_framework != inferred_framework:
        raise ValueError(
            f"Framework `{normalized_framework}` conflicts with model path `{model_path}`. "
            f"Use framework `{inferred_framework}` or remove the explicit framework."
        )
    return inferred_framework or normalized_framework or "mxq"


def split_model_paths(
    *, framework: str, model_path: str = "", mxq_path: str = "", onnx_path: str = ""
) -> tuple[str, str]:
    """Resolve generic and framework-specific local model path arguments."""

    resolved_mxq_path = mxq_path
    resolved_onnx_path = onnx_path
    if not model_path:
        return resolved_mxq_path, resolved_onnx_path
    inferred_framework = framework_from_model_path(model_path)
    if inferred_framework == "mxq":
        resolved_mxq_path = resolved_mxq_path or model_path
    elif inferred_framework == "onnx" or framework == "onnx":
        resolved_onnx_path = resolved_onnx_path or model_path
    else:
        resolved_mxq_path = resolved_mxq_path or model_path
    return resolved_mxq_path, resolved_onnx_path
