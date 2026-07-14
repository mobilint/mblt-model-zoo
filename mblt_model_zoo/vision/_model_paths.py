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
