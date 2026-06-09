"""
Wrapper classes for MBLT model execution.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from qbruntime import Cluster, CoreId

from ..utils.core_mode import CoreMode, normalize_core_mode
from ..utils.npu_backend import MobilintNPUBackend
from .utils.postprocess import build_postprocess
from .utils.preprocess import build_preprocess
from .utils.results import Results
from .utils.types import TensorLike

MODEL_CONFIG_DIR = Path(__file__).parent / "models"
SUPPORTED_FRAMEWORKS = {"mxq", "onnx"}

ONNXRUNTIME_INSTALL_GUIDE = (
    "onnxruntime is not installed. To use ONNX inference, install one of the optional extras:\n"
    "pip install mblt-model-zoo[onnxruntime]\n"
    + ("or\npip install mblt-model-zoo[onnxruntime-gpu]" if sys.platform != "darwin" else "")
)
__all__ = ["CoreMode", "normalize_core_mode", "MBLT_Engine"]


def _default_cache_dir() -> str:
    """Returns a writable cache directory for downloaded vision artifacts."""

    preferred = Path(os.path.expanduser("~/.mblt_model_zoo"))
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        test_file = preferred / ".write_test"
        test_file.touch(exist_ok=True)
        test_file.unlink(missing_ok=True)
        return str(preferred)
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "mblt_model_zoo"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)


MOBILINT_CACHE_DIR = _default_cache_dir()


def _load_onnxruntime() -> Any:
    """Loads ``onnxruntime`` only when ONNX inference is requested.

    Returns:
        The imported ``onnxruntime`` module.

    Raises:
        ImportError: If ``onnxruntime`` is unavailable in the current environment.
    """

    try:
        module = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise ImportError(ONNXRUNTIME_INSTALL_GUIDE) from exc

    if not hasattr(module, "InferenceSession"):
        module_path = getattr(module, "__file__", None) or "<namespace package>"
        msg = (
            "onnxruntime is installed, but the package is incomplete or broken and does not expose "
            f"`InferenceSession` (resolved from {module_path}). "
            f"{ONNXRUNTIME_INSTALL_GUIDE.replace('is not installed. To use ONNX inference, install', 'Reinstall')}"
        )
        raise ImportError(msg)

    return module


def _resolve_onnx_providers(ort_module: Any, requested_providers: Sequence[str] | None = None) -> list[str]:
    """Selects ONNX Runtime execution providers.

    Args:
        ort_module: Imported ``onnxruntime`` module or compatible test double.
        requested_providers: Optional provider order requested by the caller.

    Returns:
        The provider list passed to ``InferenceSession``.
    """

    if requested_providers is not None:
        return list(requested_providers)

    get_available = getattr(ort_module, "get_available_providers", None)
    if not callable(get_available):
        return ["CPUExecutionProvider"]

    available_providers = set(get_available())
    preferred_providers = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    resolved_providers = [provider for provider in preferred_providers if provider in available_providers]
    return resolved_providers or ["CPUExecutionProvider"]


def _framework_from_model_path(model_path: str) -> str | None:
    """Infers the runtime framework from a local model path suffix."""

    suffix = Path(model_path).suffix.lower()
    if suffix == ".mxq":
        return "mxq"
    if suffix == ".onnx":
        return "onnx"
    return None


def _resolve_framework(framework: str | None, model_path: str = "") -> str:
    """Resolves the execution framework from explicit input and model path."""

    normalized_framework = framework.lower() if framework is not None else None
    if normalized_framework is not None and normalized_framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework: {framework}. Must be one of {sorted(SUPPORTED_FRAMEWORKS)}.")

    inferred_framework = _framework_from_model_path(model_path) if model_path else None
    if normalized_framework and inferred_framework and normalized_framework != inferred_framework:
        raise ValueError(
            f"Framework `{normalized_framework}` conflicts with model path `{model_path}`. "
            f"Use framework `{inferred_framework}` or remove the explicit framework."
        )

    return inferred_framework or normalized_framework or "mxq"


def _split_model_paths(
    *,
    framework: str,
    model_path: str = "",
    mxq_path: str = "",
    onnx_path: str = "",
) -> tuple[str, str]:
    """Resolves generic and framework-specific local model path arguments."""

    resolved_mxq_path = mxq_path
    resolved_onnx_path = onnx_path
    if not model_path:
        return resolved_mxq_path, resolved_onnx_path

    inferred_framework = _framework_from_model_path(model_path)
    if inferred_framework == "mxq":
        resolved_mxq_path = resolved_mxq_path or model_path
    elif inferred_framework == "onnx":
        resolved_onnx_path = resolved_onnx_path or model_path
    elif framework == "onnx":
        resolved_onnx_path = resolved_onnx_path or model_path
    else:
        resolved_mxq_path = resolved_mxq_path or model_path

    return resolved_mxq_path, resolved_onnx_path


class MBLT_Engine:
    """Main engine class for running vision models from the MBLT zoo.

    Handles the full pipeline: Preprocessing -> Inference -> Postprocessing.

    Attributes:
            file_cfg: Model configuration.
            pre_cfg: Preprocessing configuration.
            post_cfg: Postprocessing configuration.
            model: The underlying MXQ_Model.
            device: The torch device being used.
    """

    def __init__(
        self,
        model_cls: str | dict[str, Any],
        model_type: str = "DEFAULT",
        model_path: str = "",
        mxq_path: str = "",
        onnx_path: str = "",
        dev_no: int | None = None,
        core_mode: CoreMode | None = None,
        target_cores: Sequence[str | CoreId] | None = None,
        target_clusters: Sequence[int | Cluster] | None = None,
        postprocess_kwargs: dict[str, Any] | None = None,
        framework: str | None = None,
        onnx_providers: Sequence[str] | None = None,
    ) -> None:
        """Initializes the MBLT_Engine.

        Args:
            model_cls(if dict):
                file_cfg: Model configuration.
                    model_path: generic path to local model file
                    mxq_path: path to mxq file
                    onnx_path: path to onnx file
                    dev_no: Accelerator No.
                    core_mode: single, multi, global4, global8
                    target_cores: single mode
                    target_clusters: multi, global modes
                pre_cfg: Preprocessing configuration.
                post_cfg: Postprocessing configuration.
            model_cls(not dict): model name or yaml path
            postprocess_kwargs: Optional runtime overrides passed to the postprocessor builder.
            framework: Execution framework, either "mxq" or "onnx". When omitted,
                ``model_path`` suffix is used first, then MXQ is the fallback.
            onnx_providers: Optional ONNX Runtime execution provider order.
        """

        if isinstance(model_cls, dict):  # direct setting
            model_config_part = copy.deepcopy(model_cls)
        else:  # setting via yaml file path or model name
            config_path = model_cls
            if not os.path.isfile(config_path):
                config_path = self.model_name_aliasing(config_path)
                config_path = os.path.join(MODEL_CONFIG_DIR, config_path)

            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            if not isinstance(full_config, dict):
                raise TypeError(f"Model configuration '{config_path}' should define a dictionary.")

            model_config_part = full_config.get(model_type)
            if model_config_part is None:
                raise ValueError(f"Model type '{model_type}' not found in configuration.")

            if isinstance(model_config_part, str):  # Resolve alias
                resolved_config = full_config.get(model_config_part)
                if resolved_config is None:
                    raise ValueError(f"Model alias '{model_config_part}' not found in configuration.")
                model_config_part = resolved_config

            if not isinstance(model_config_part, dict):
                raise TypeError(f"Resolved model configuration for '{model_type}' is not a dictionary.")

            if "update" in model_config_part:
                base_config_key = model_config_part.pop("update")
                base_config = full_config.get(base_config_key)
                if base_config is None:
                    raise ValueError(f"Base configuration '{base_config_key}' not found for update.")

                merged_config = copy.deepcopy(base_config)
                for key, value in model_config_part.items():
                    if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                        merged_config[key].update(value)
                    else:
                        merged_config[key] = value
                model_config_part = merged_config

        file_cfg_model_path = str(model_config_part["file_cfg"].get("model_path", ""))
        framework_model_path = model_path or file_cfg_model_path
        self.framework = _resolve_framework(framework, framework_model_path)
        mxq_path, onnx_path = _split_model_paths(
            framework=self.framework,
            model_path=model_path,
            mxq_path=mxq_path,
            onnx_path=onnx_path,
        )

        _mxq_path_passed = bool(mxq_path)
        _onnx_path_passed = bool(onnx_path)
        _dev_no_passed = dev_no is not None
        _core_mode_passed = core_mode is not None
        _target_cores_passed = target_cores is not None
        _target_clusters_passed = target_clusters is not None

        if dev_no is None:
            dev_no = 0
        if core_mode is None:
            core_mode = "single"
        if target_cores is None:
            target_cores = ["0:0", "0:1", "0:2", "0:3", "1:0", "1:1", "1:2", "1:3"]
        if target_clusters is None:
            target_clusters = [0, 1]

        self.file_cfg = copy.deepcopy(model_config_part["file_cfg"])
        file_cfg_model_path = self.file_cfg.pop("model_path", "")
        if file_cfg_model_path:
            yaml_mxq_path, yaml_onnx_path = _split_model_paths(
                framework=self.framework,
                model_path=file_cfg_model_path,
                mxq_path=str(self.file_cfg.get("mxq_path", "")),
                onnx_path=str(self.file_cfg.get("onnx_path", "")),
            )
            self.file_cfg["mxq_path"] = yaml_mxq_path
            self.file_cfg["onnx_path"] = yaml_onnx_path
        if _mxq_path_passed or "mxq_path" not in self.file_cfg:
            self.file_cfg["mxq_path"] = mxq_path
        if _onnx_path_passed or "onnx_path" not in self.file_cfg:
            self.file_cfg["onnx_path"] = onnx_path
        if _core_mode_passed or "core_mode" not in self.file_cfg:
            self.file_cfg["core_mode"] = core_mode
        if _target_cores_passed or "target_cores" not in self.file_cfg:
            self.file_cfg["target_cores"] = target_cores
        if _target_clusters_passed or "target_clusters" not in self.file_cfg:
            self.file_cfg["target_clusters"] = target_clusters
        if _dev_no_passed or "dev_no" not in self.file_cfg:
            self.file_cfg["dev_no"] = dev_no

        self.pre_cfg = copy.deepcopy(model_config_part["pre_cfg"])
        self.post_cfg = copy.deepcopy(model_config_part["post_cfg"])
        self.postprocess_kwargs = {} if postprocess_kwargs is None else dict(postprocess_kwargs)
        self.file_config_cleansing()

        self.model: Any
        self._mxq_model: MobilintNPUBackend | None = None
        self._onnx_session: Any = None

        if self.framework == "onnx":
            ort = _load_onnxruntime()
            resolved_onnx_path = self.file_cfg.get("onnx_path")
            if not resolved_onnx_path:
                raise RuntimeError(
                    f"ONNX path not resolved for model {model_cls}. Make sure the model repository has an ONNX file."
                )
            if not os.path.isfile(resolved_onnx_path):
                raise FileNotFoundError(f"ONNX file not found at: {resolved_onnx_path}")

            providers = _resolve_onnx_providers(ort, onnx_providers)
            self._onnx_session = ort.InferenceSession(resolved_onnx_path, providers=providers)
            self.model = self._onnx_session
            self.input_name = self._onnx_session.get_inputs()[0].name
            self.output_names = [o.name for o in self._onnx_session.get_outputs()]
        else:
            self._mxq_model = MobilintNPUBackend(**self._mxq_backend_kwargs())
            self.model = self._mxq_model
            self._mxq_model.create()
            self._mxq_model.launch()

            if self._mxq_model.get_dtype() == "DataType.Uint8":
                self.pre_cfg.pop("Normalize", None)

        self.preprocessor = build_preprocess(self.pre_cfg)
        self.postprocessor = build_postprocess(self.pre_cfg, self.post_cfg, **self.postprocess_kwargs)
        self.device = torch.device("cpu")

    def _mxq_backend_kwargs(self) -> dict[str, Any]:
        """Builds the MXQ backend kwargs from the resolved file config."""

        excluded_keys = {"repo_id", "filename", "revision", "onnx_filename", "onnx_path"}
        return {key: value for key, value in self.file_cfg.items() if key not in excluded_keys}

    def _derive_onnx_filename(self) -> str | None:
        """Returns the ONNX filename associated with the configured MXQ artifact."""

        onnx_filename = self.file_cfg.get("onnx_filename")
        if onnx_filename:
            return onnx_filename

        filename = self.file_cfg.get("filename")
        if not filename:
            return None

        derived_name = f"{Path(filename).stem}.onnx"
        self.file_cfg["onnx_filename"] = derived_name
        return derived_name

    def _resolve_local_onnx_path(self, mxq_path: str) -> str | None:
        """Tries to resolve a sibling ONNX file next to a local MXQ artifact."""

        onnx_path = self.file_cfg.get("onnx_path", "")
        if onnx_path and os.path.isfile(onnx_path):
            return onnx_path

        onnx_filename = self._derive_onnx_filename()
        if onnx_filename:
            sibling_path = Path(mxq_path).with_name(onnx_filename)
            if sibling_path.is_file():
                return str(sibling_path)

        if mxq_path.endswith(".mxq"):
            suffix_swapped = f"{mxq_path[:-4]}.onnx"
            if os.path.isfile(suffix_swapped):
                return suffix_swapped

        return None

    def _download_hub_artifact(
        self,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        subfolders: Sequence[str] | None = None,
    ) -> str:
        """Downloads a model artifact from Hugging Face Hub and returns its cache path."""

        last_error: Exception | None = None
        normalized_subfolders = [""] if subfolders is None else list(subfolders)
        for subfolder in normalized_subfolders:
            kwargs: dict[str, Any] = {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "local_dir": MOBILINT_CACHE_DIR,
            }
            if subfolder:
                kwargs["subfolder"] = subfolder
            try:
                return hf_hub_download(**kwargs)
            except EntryNotFoundError as exc:
                last_error = exc

        attempted_paths = ", ".join(
            f"{subfolder}/{filename}" if subfolder else filename for subfolder in normalized_subfolders
        )
        raise RuntimeError(
            f"Failed to download model from HuggingFace. Tried repo '{repo_id}' at: {attempted_paths}."
        ) from last_error

    def file_config_cleansing(self) -> None:
        """Validates and resolves the MXQ and ONNX model file paths in ``self.file_cfg``."""
        framework = getattr(self, "framework", "mxq")
        model_path = self.file_cfg.pop("model_path", "")
        if model_path:
            mxq_path, onnx_path = _split_model_paths(
                framework=framework,
                model_path=model_path,
                mxq_path=str(self.file_cfg.get("mxq_path", "")),
                onnx_path=str(self.file_cfg.get("onnx_path", "")),
            )
            self.file_cfg["mxq_path"] = mxq_path
            self.file_cfg["onnx_path"] = onnx_path
        mxq_path = self.file_cfg.get("mxq_path", "")
        onnx_path = self.file_cfg.get("onnx_path", "")
        onnx_filename = self._derive_onnx_filename()

        if onnx_path and os.path.isfile(onnx_path):
            self.file_cfg["onnx_path"] = onnx_path
            if framework == "onnx":
                return

        if mxq_path and os.path.isfile(mxq_path):
            resolved_local_onnx = self._resolve_local_onnx_path(mxq_path)
            if resolved_local_onnx is not None:
                self.file_cfg["onnx_path"] = resolved_local_onnx
                self.file_cfg.pop("repo_id", None)
                self.file_cfg.pop("filename", None)
                self.file_cfg.pop("revision", None)
            elif framework == "mxq":
                self.file_cfg.pop("repo_id", None)
                self.file_cfg.pop("filename", None)
                self.file_cfg.pop("revision", None)
            if framework == "mxq":
                return

        repo_id = self.file_cfg.pop("repo_id", None)
        filename = self.file_cfg.pop("filename", None)
        revision = self.file_cfg.pop("revision", None)
        if not repo_id or not revision:
            return

        if filename and framework == "mxq":
            core_mode = self.file_cfg.get("core_mode")
            subfolders = ["aries", f"aries/{core_mode}"] if core_mode else ["aries"]
            self.file_cfg["mxq_path"] = self._download_hub_artifact(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                subfolders=subfolders,
            )

        if onnx_filename and framework == "onnx" and not self.file_cfg.get("onnx_path"):
            self.file_cfg["onnx_path"] = self._download_hub_artifact(
                repo_id=repo_id,
                filename=onnx_filename,
                revision=revision,
            )

    def _prepare_onnx_inputs(self, x: TensorLike) -> dict[str, np.ndarray]:
        """Normalizes runtime inputs to match the ONNX session contract."""

        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            x_np = x
        else:
            raise TypeError(f"Got unexpected type for ONNX input x={type(x)}.")

        if x_np.dtype == np.float64:
            x_np = x_np.astype(np.float32)

        expected_shape = self._require_onnx_session().get_inputs()[0].shape
        if len(expected_shape) == 4:
            if x_np.ndim == 3:
                first_channel_count = x_np.shape[0]
                last_channel_count = x_np.shape[-1]
            elif x_np.ndim == 4:
                first_channel_count = x_np.shape[1]
                last_channel_count = x_np.shape[-1]
            else:
                first_channel_count = None
                last_channel_count = None

            expected_second_dim = expected_shape[1]
            expected_last_dim = expected_shape[-1]
            second_matches_channels = isinstance(expected_second_dim, int) and (
                expected_second_dim == first_channel_count or expected_second_dim == last_channel_count
            )
            last_matches_channels = isinstance(expected_last_dim, int) and (
                expected_last_dim == first_channel_count or expected_last_dim == last_channel_count
            )

            expected_layout = None
            expected_channels = None
            if second_matches_channels and not last_matches_channels:
                expected_layout = "nchw"
                expected_channels = expected_second_dim
            elif last_matches_channels and not second_matches_channels:
                expected_layout = "nhwc"
                expected_channels = expected_last_dim
            elif isinstance(expected_second_dim, int) and expected_second_dim in {1, 2, 3, 4}:
                expected_layout = "nchw"
                expected_channels = expected_second_dim
            elif isinstance(expected_last_dim, int) and expected_last_dim in {1, 2, 3, 4}:
                expected_layout = "nhwc"
                expected_channels = expected_last_dim

            if x_np.ndim == 3:
                if expected_layout == "nchw" and expected_channels is not None and x_np.shape[0] == expected_channels:
                    x_np = np.expand_dims(x_np, axis=0)
                elif (
                    expected_layout == "nchw" and expected_channels is not None and x_np.shape[-1] == expected_channels
                ):
                    x_np = np.transpose(x_np, (2, 0, 1))
                    x_np = np.expand_dims(x_np, axis=0)
                elif (
                    expected_layout == "nhwc" and expected_channels is not None and x_np.shape[-1] == expected_channels
                ):
                    x_np = np.expand_dims(x_np, axis=0)
                elif expected_layout == "nhwc" and expected_channels is not None and x_np.shape[0] == expected_channels:
                    x_np = np.transpose(x_np, (1, 2, 0))
                    x_np = np.expand_dims(x_np, axis=0)
            elif x_np.ndim == 4:
                if expected_layout == "nchw" and expected_channels is not None:
                    if x_np.shape[1] == expected_channels:
                        pass
                    elif x_np.shape[-1] == expected_channels:
                        x_np = np.transpose(x_np, (0, 3, 1, 2))
                elif expected_layout == "nhwc" and expected_channels is not None:
                    if x_np.shape[-1] == expected_channels:
                        pass
                    elif x_np.shape[1] == expected_channels:
                        x_np = np.transpose(x_np, (0, 2, 3, 1))

        return {self.input_name: x_np}

    def _require_onnx_session(self) -> Any:
        """Return the active ONNX session."""

        session = getattr(self, "_onnx_session", None)
        if session is None:
            fallback = getattr(self, "model", None)
            if fallback is not None and hasattr(fallback, "get_inputs") and hasattr(fallback, "run"):
                return fallback
            raise RuntimeError("ONNX session is not initialized.")
        return session

    def _require_mxq_model(self) -> MobilintNPUBackend:
        """Return the active MXQ backend."""

        model = getattr(self, "_mxq_model", None)
        if model is None:
            fallback = getattr(self, "model", None)
            if fallback is not None and hasattr(fallback, "create") and hasattr(fallback, "launch"):
                return cast(MobilintNPUBackend, fallback)
            raise RuntimeError("MXQ backend is not initialized.")
        return model

    def __call__(
        self,
        x: TensorLike,
    ) -> Any:
        """Runs raw model inference on the input.

        Note:
                This does NOT include preprocessing or postprocessing.

        Args:
                x: Input tensor for the model.

        Returns:
                Raw model output.
        """
        if self.framework == "onnx":
            outputs = self._require_onnx_session().run(self.output_names, self._prepare_onnx_inputs(x))
            if len(outputs) == 1:
                return outputs[0]
            return outputs
        return cast(Any, self._require_mxq_model())(x)

    def preprocess(
        self,
        x: Any,
        **kwargs: Any,
    ) -> Any:
        """Runs preprocessing on the input.

        Args:
                x: Input data.
                **kwargs: Additional arguments for preprocessing.

        Returns:
                Preprocessed data.
        """
        return self.preprocessor(x, **kwargs)

    def preprocess_with_metadata(
        self,
        x: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Runs preprocessing and returns metadata needed for exact postprocess scaling.

        Args:
                x: Input data.

        Returns:
                A tuple of preprocessed data and metadata such as ``ratio_pad``.
        """
        return self.preprocessor.with_metadata(x)

    def postprocess(
        self,
        x: Any,
        **kwargs: Any,
    ) -> Results:
        """Runs postprocessing on the input.

        Args:
                x: Input data.
                **kwargs: Additional arguments for postprocessing.

        Returns:
                Postprocessed results.
        """
        pre_result = self.postprocessor(x, **kwargs)
        result_kwargs = dict(kwargs)
        conf_thres = getattr(self.postprocessor, "conf_thres", None)
        iou_thres = getattr(self.postprocessor, "iou_thres", None)
        if "conf_thres" not in result_kwargs and conf_thres is not None:
            result_kwargs["conf_thres"] = conf_thres
        if "iou_thres" not in result_kwargs and iou_thres is not None:
            result_kwargs["iou_thres"] = iou_thres
        return Results(self.pre_cfg, self.post_cfg, pre_result, **result_kwargs)

    def set_postprocess_thresholds(self, conf_thres: float | None = None, iou_thres: float | None = None) -> None:
        """Updates configurable postprocess thresholds for the current model.

        Args:
                conf_thres: Optional confidence threshold override.
                iou_thres: Optional IoU threshold override.

        Raises:
                NotImplementedError: If the current postprocessor does not support thresholds.
        """
        set_threshold = getattr(self.postprocessor, "set_threshold", None)
        if set_threshold is None:
            raise NotImplementedError(
                f"Threshold overrides are not supported for task `{self.post_cfg.get('task', 'unknown')}`."
            )
        set_threshold(conf_thres=conf_thres, iou_thres=iou_thres)

    def to(
        self,
        device: str | torch.device,
    ) -> None:
        """Moves the engine and its components to the specified device.

        Args:
                device: Target device.

        Raises:
                TypeError: If device type is unexpected.
        """
        self.preprocessor.to(device)
        self.postprocessor.to(device)

        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(f"Got unexpected type for device={type(device)}.")

    def cpu(self) -> None:
        """Moves the engine to CPU."""
        self.to(device="cpu")

    def gpu(self) -> None:
        """Moves the engine to GPU (CUDA)."""
        self.to(device="cuda")

    def cuda(
        self,
        device: str | int = 0,
    ) -> None:
        """Moves the engine to CUDA device.

        Args:
                device: CUDA device identifier. Defaults to 0.

        Raises:
                ValueError: If device string is invalid.
                RuntimeError: If CUDA is not available.
        """
        if isinstance(device, int):
            device = f"cuda:{device}"
        elif isinstance(device, str):
            if not device.startswith("cuda:"):
                raise ValueError("Invalid device string. It should start with 'cuda:'.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your environment.")
        self.to(device=device)

    def launch(self) -> None:
        """Launches the underlying model."""
        if self.framework == "mxq":
            self._require_mxq_model().launch()

    def dispose(self) -> None:
        """Disposes the underlying model."""
        if self.framework == "mxq":
            self._require_mxq_model().dispose()

    def model_name_aliasing(self, model_name: str) -> str:
        """Finds the YAML config filename that matches the given model name.

        Matching is case-insensitive and first preserves separator boundaries
        so names such as ``regnet_x_16gf`` do not collide with
        ``regnet_x_1_6gf``. A separator-stripped fallback is used only when it
        resolves to a single unique configuration, so inputs like ``resnet50``,
        ``ResNet50``, ``Resnet_50``, and ``resnet-50`` all resolve to
        ``ResNet50.yaml``.

        Args:
            model_name: The model identifier provided by the caller.

        Returns:
            The exact YAML filename (basename only) stored in
            ``MODEL_CONFIG_DIR`` that corresponds to ``model_name``.

        Raises:
            ValueError: If no YAML file matches, or if the name is ambiguous
                (i.e., multiple files match after normalization).
        """

        def _stem(name: str) -> str:
            """Returns a YAML filename stem when a YAML suffix is present."""
            return name[: -len(".yaml")] if name.lower().endswith(".yaml") else name

        def _normalize_separators(name: str) -> str:
            """Normalizes separator style while preserving separator boundaries."""
            return "_".join(part for part in _stem(name).replace("-", "_").replace(" ", "_").lower().split("_") if part)

        def _normalize_compact(name: str) -> str:
            """Strips separators and lowercases a model name for fallback matching."""
            return _stem(name).replace("_", "").replace("-", "").replace(" ", "").lower()

        candidates = [p.name for p in MODEL_CONFIG_DIR.glob("*.yaml")]
        candidate_stems = {name: name[: -len(".yaml")] for name in candidates}

        normalized_separator_input = _normalize_separators(model_name)
        separator_matches = [
            name for name, stem in candidate_stems.items() if _normalize_separators(stem) == normalized_separator_input
        ]
        if len(separator_matches) == 1:
            return separator_matches[0]

        normalized_compact_input = _normalize_compact(model_name)
        matches = [
            name for name, stem in candidate_stems.items() if _normalize_compact(stem) == normalized_compact_input
        ]

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise ValueError(
                f"Model name '{model_name}' is ambiguous. "
                f"It matches multiple configurations: {sorted(matches)}. "
                "Please use a more specific name."
            )
        raise ValueError(
            f"No model configuration found for '{model_name}'. "
            f"Available models are located in '{MODEL_CONFIG_DIR}'. "
            "Check spelling or use the exact YAML filename."
        )
