"""
Wrapper classes for MBLT model execution.
"""

import copy  # Added import
import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
import yaml
from huggingface_hub import hf_hub_download
from qbruntime import Cluster, CoreId

from ..utils.npu_backend import MobilintNPUBackend
from .utils.postprocess import build_postprocess
from .utils.preprocess import build_preprocess
from .utils.results import Results
from .utils.types import TensorLike

MODEL_CONFIG_DIR = Path(__file__).parent / "models"
MOBILINT_CACHE_DIR = os.path.expanduser("~/.mblt_model_zoo")


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
        model_cls: Union[str, dict],
        model_type: str = "DEFAULT",
        mxq_path: str = "",
        dev_no: int = 0,
        core_mode: Literal["single", "multi", "global4", "global8"] = "single",
        target_cores: Optional[List[Union[str, CoreId]]] = None,
        target_clusters: Optional[List[Union[int, Cluster]]] = None,
    ):
        """Initializes the MBLT_Engine.

        Args:
            model_cls(if dict):
                file_cfg: Model configuration.
                    mxq_path: path to mxq file
                    dev_no: Accelerator No.
                    core_mode: single, multi, global4, global8
                    target_cores: single mode
                    target_clusters: multi, global modes
                pre_cfg: Preprocessing configuration.
                post_cfg: Postprocessing configuration.
            model_cls(not dict): model name or yaml path
        """

        if target_cores is None:
            target_cores = ["0:0", "0:1", "0:2", "0:3", "1:0", "1:1", "1:2", "1:3"]
        if target_clusters is None:
            target_clusters = [0, 1]

        if isinstance(model_cls, dict):  # direct setting
            self.file_cfg = model_cls["file_cfg"]
            self.pre_cfg = model_cls["pre_cfg"]
            self.post_cfg = model_cls["post_cfg"]
        else:  # setting via yaml file path or model name
            config_path = model_cls
            if not os.path.isfile(config_path):
                config_path = self.model_name_aliasing(config_path)
                config_path = os.path.join(MODEL_CONFIG_DIR, config_path)

            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)

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

            self.file_cfg = model_config_part["file_cfg"]
            self.file_cfg["mxq_path"] = mxq_path
            self.file_cfg["core_mode"] = core_mode
            self.file_cfg["target_cores"] = target_cores
            self.file_cfg["target_clusters"] = target_clusters
            self.file_config_cleansing()

        # These assignments are now correctly placed after ensuring model_config_part is a dict
        self.pre_cfg = model_config_part["pre_cfg"]
        self.post_cfg = model_config_part["post_cfg"]

        self.model = MobilintNPUBackend(dev_no=dev_no, **self.file_cfg)
        self.model.create()
        self.model.launch()

        if self.model.get_dtype() == "DataType.Uint8":
            self.pre_cfg.pop("Normalize", None)

        self.preprocessor = build_preprocess(self.pre_cfg)
        self.postprocessor = build_postprocess(self.pre_cfg, self.post_cfg)
        self.device = torch.device("cpu")

    def file_config_cleansing(self):
        """Validates and resolves the MXQ model file path in ``self.file_cfg``.

        If ``self.file_cfg["mxq_path"]`` already points to an existing file,
        the Hub-related keys (``repo_id``, ``filename``, ``revision``) are
        removed from the config as they are no longer needed.

        Otherwise the method downloads the model from HuggingFace Hub using
        those keys and updates ``self.file_cfg["mxq_path"]`` with the local
        cache path.

        Raises:
            RuntimeError: If the model cannot be downloaded from HuggingFace
                Hub, wrapping the original exception for full traceback context.
        """
        if os.path.exists(self.file_cfg["mxq_path"]):
            self.file_cfg.pop("repo_id")
            self.file_cfg.pop("filename")
            self.file_cfg.pop("revision")
        else:
            try:
                cached_file = hf_hub_download(
                    repo_id=self.file_cfg.pop("repo_id"),
                    filename=self.file_cfg.pop("filename"),
                    subfolder=f"aries/{self.file_cfg['core_mode']}",
                    revision=self.file_cfg.pop("revision"),
                    local_dir=MOBILINT_CACHE_DIR,
                )
                self.file_cfg["mxq_path"] = cached_file
            except Exception as e:
                raise RuntimeError("Failed to download model from HuggingFace") from e

    def __call__(
        self,
        x: TensorLike,
    ):
        """Runs raw model inference on the input.

        Note:
                This does NOT include preprocessing or postprocessing.

        Args:
                x: Input tensor for the model.

        Returns:
                Raw model output.
        """
        return self.model(x)

    def preprocess(
        self,
        x,
        **kwargs,
    ):
        """Runs preprocessing on the input.

        Args:
                x: Input data.
                **kwargs: Additional arguments for preprocessing.

        Returns:
                Preprocessed data.
        """
        return self.preprocessor(x, **kwargs)

    def postprocess(
        self,
        x,
        **kwargs,
    ) -> Results:
        """Runs postprocessing on the input.

        Args:
                x: Input data.
                **kwargs: Additional arguments for postprocessing.

        Returns:
                Postprocessed results.
        """
        pre_result = self.postprocessor(x, **kwargs)
        return Results(self.pre_cfg, self.post_cfg, pre_result, **kwargs)

    def to(
        self,
        device: Union[str, torch.device],
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
        device: Union[str, int] = 0,
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
        self.model.launch()

    def dispose(self) -> None:
        """Disposes the underlying model."""
        self.model.dispose()

    def model_name_aliasing(self, model_name: str) -> str:
        """Finds the YAML config filename that matches the given model name.

        Matching is case-insensitive and ignores separator characters (``_``,
        ``-``, and spaces), so inputs like ``resnet50``, ``ResNet50``,
        ``Resnet_50``, and ``resnet-50`` all resolve to ``ResNet50.yaml``.

        Args:
            model_name: The model identifier provided by the caller.

        Returns:
            The exact YAML filename (basename only) stored in
            ``MODEL_CONFIG_DIR`` that corresponds to ``model_name``.

        Raises:
            ValueError: If no YAML file matches, or if the name is ambiguous
                (i.e., multiple files match after normalization).
        """

        def _normalize(name: str) -> str:
            """Strips separators and lowercases a model name for comparison."""
            return name.replace("_", "").replace("-", "").replace(" ", "").lower()

        normalized_input = _normalize(model_name)

        candidates = [p.name for p in MODEL_CONFIG_DIR.glob("*.yaml")]
        matches = [name for name in candidates if _normalize(name[: -len(".yaml")]) == normalized_input]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"Model name '{model_name}' is ambiguous. "
                f"It matches multiple configurations: {sorted(matches)}. "
                "Please use a more specific name."
            )
        else:
            raise ValueError(
                f"No model configuration found for '{model_name}'. "
                f"Available models are located in '{MODEL_CONFIG_DIR}'. "
                "Check spelling or use the exact YAML filename."
            )
