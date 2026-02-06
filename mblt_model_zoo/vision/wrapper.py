"""
Wrapper classes for MBLT model execution.
"""

import os
from typing import Union

import numpy as np
import qbruntime
import torch
from huggingface_hub import hf_hub_download

from ..utils.logging import log_model_details
from .utils.postprocess import build_postprocess
from .utils.preprocess import build_preprocess
from .utils.results import Results
from .utils.types import ModelInfoSet, TensorLike


class MBLT_Engine:
    """
    Main engine class for running vision models from the MBLT zoo.
    Handles the full pipeline: Preprocessing -> Inference -> Postprocessing.
    """

    def __init__(self, model_cfg: dict, pre_cfg: dict, post_cfg: dict):
        """Initializes the MBLT_Engine.

        Args:
            model_cfg (dict): Model configuration.
            pre_cfg (dict): Preprocessing configuration.
            post_cfg (dict): Postprocessing configuration.
        """
        self.model_cfg = model_cfg
        self.pre_cfg = pre_cfg
        self.post_cfg = post_cfg
        self.model = MXQ_Model(**self.model_cfg)
        self._preprocess = build_preprocess(self.pre_cfg)
        self._postprocess = build_postprocess(self.pre_cfg, self.post_cfg)
        self.device = torch.device("cpu")

    @staticmethod
    def _get_configs(
        model_info_set: ModelInfoSet,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global8",
        product: str = "aries",
    ):
        """Extracts and prepares configuration dictionaries from a ModelInfoSet.

        Args:
            model_info_set (ModelInfoSet): Set of model configurations.
            local_path (str, optional): Path to a local model file. Defaults to None.
            model_type (str, optional): Model configuration type. Defaults to "DEFAULT".
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".

        Returns:
            tuple: (model_cfg, pre_cfg, post_cfg) dictionaries.
        """
        assert (
            model_type in model_info_set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {model_info_set.__dict__.keys()}"

        # Use copy() to avoid modifying the original ModelInfo in-place
        model_cfg = model_info_set.__dict__[model_type].value.model_cfg.copy()
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product

        pre_cfg = model_info_set.__dict__[model_type].value.pre_cfg.copy()
        post_cfg = model_info_set.__dict__[model_type].value.post_cfg.copy()

        return model_cfg, pre_cfg, post_cfg

    def __call__(self, x: TensorLike):
        """Runs raw model inference on the input.

        Note:
            This does NOT include preprocessing or postprocessing.

        Args:
            x (TensorLike): Input tensor for the model.

        Returns:
            Any: Raw model output.
        """
        return self.model(x)

    def preprocess(self, x, **kwargs):
        """Runs preprocessing on the input.

        Args:
            x: Input data.
            **kwargs: Additional arguments for preprocessing.

        Returns:
            Any: Preprocessed data.
        """
        return self._preprocess(x, **kwargs)

    def postprocess(self, x, **kwargs) -> Results:
        """Runs postprocessing on the input.

        Args:
            x: Input data.
            **kwargs: Additional arguments for postprocessing.

        Returns:
            Results: Postprocessed results.
        """
        pre_result = self._postprocess(x, **kwargs)
        return Results(self.pre_cfg, self.post_cfg, pre_result, **kwargs)

    def to(self, device: Union[str, torch.device]):
        """Moves the engine and its components to the specified device.

        Args:
            device (Union[str, torch.device]): Target device.

        Raises:
            TypeError: If device type is unexpected.
        """
        self._preprocess.to(device)
        self._postprocess.to(device)

        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(f"Got unexpected type for device={type(device)}.")

    def cpu(self):
        """Moves the engine to CPU."""
        self.to(device="cpu")

    def gpu(self):
        """Moves the engine to GPU (CUDA)."""
        self.to(device="cuda")

    def cuda(self, device: Union[str, int] = 0):
        """Moves the engine to CUDA device.

        Args:
            device (Union[str, int], optional): CUDA device identifier. Defaults to 0.

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

    def launch(self):
        """Launches the underlying model."""
        self.model.launch()

    def dispose(self):
        """Disposes the underlying model."""
        self.model.dispose()


class MXQ_Model:
    """
    Low-level wrapper for managing model execution on the Mobilint MXQ accelerator.
    Handles communication with the qbruntime and resource management.
    """

    def __init__(
        self,
        product: str = "aries",
        infer_mode: str = "global8",
        repo_id: str = None,
        filename: str = None,
        local_path: str = None,
    ):
        """Initializes the MXQ_Model.

        Args:
            repo_id (str, optional): Hugging Face repository ID. Defaults to None.
            filename (str, optional): Model filename. Defaults to None.
            local_path (str, optional): Path to local model file. Defaults to None.
            infer_mode (str, optional): Inference execution mode. Defaults to "global8".
            product (str, optional): Target hardware product. Defaults to "aries".

        Raises:
            ValueError: If inference mode or product is invalid, or model file is missing.
        """
        self.infer_mode = infer_mode
        assert infer_mode in [
            "single",
            "multi",
            "global4",
            "global8",
        ], "Inappropriate inference mode"

        self.product = product
        assert product in ["aries", "regulus"], "Inappropriate product"

        self.acc = qbruntime.Accelerator()
        # ----------------Core Allocation-------------------------
        mc = qbruntime.ModelConfig()
        if self.product == "aries":
            if self.infer_mode == "single":
                pass  # default is single with all cores
            elif self.infer_mode == "multi":
                mc.set_multi_core_mode(
                    [qbruntime.Cluster.Cluster0, qbruntime.Cluster.Cluster1]
                )
            elif self.infer_mode == "global4":
                mc.set_global4_core_mode(
                    [qbruntime.Cluster.Cluster0, qbruntime.Cluster.Cluster1]
                )
            elif self.infer_mode == "global8":
                mc.set_global8_core_mode()
            else:
                raise ValueError("Inappropriate inference mode")
        elif self.product == "regulus":
            assert (
                self.infer_mode == "single"
            ), "Only single core mode is available on REGULUS"
        else:
            raise ValueError("Inappropriate product")

        # -----------------Model Preparation-----------------------
        if local_path is not None:
            assert os.path.isfile(
                local_path
            ), "The model should be prepared on local path"
            cached_file = local_path
        elif repo_id is not None and filename is not None:
            try:
                cached_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=f"{self.product}/{self.infer_mode}",
                    local_dir=os.path.expanduser("~/.mblt_model_zoo/vision"),
                )
            except Exception as e:
                print(e)
                print("Failed to download model from HuggingFace")
        else:
            raise ValueError("The model should be prepared on server or local path")

        # ----------------Initialize Model----------------------
        self.model = qbruntime.Model(cached_file, mc)
        log_model_details(cached_file)
        self.model.launch(self.acc)

    def __call__(self, x: TensorLike):
        """Runs inference on the input using the accelerator.

        Args:
            x (TensorLike): Input tensor or array.

        Returns:
            Any: NPU outputs.
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        assert isinstance(x, np.ndarray), "Input should be a numpy array"
        npu_outs = self.model.infer(x)
        return npu_outs

    def launch(self):
        """Launch the model on the accelerator."""
        self.model.launch(self.acc)

    def dispose(self):
        """Dispose the model and free resources."""
        self.model.dispose()
